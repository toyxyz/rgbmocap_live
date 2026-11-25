"""VMC 2.0 body sender for the normalized Unity humanoid used by Warudo."""
from ipaddress import IPv4Address
import socket
from struct import Struct
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

from .bone_stream import Y_TO_Z
from .live_pose import FINGERS, NAMES, PARENTS
from .mixamo_rig import align


VMC_HOST = '127.0.0.1'
VMC_PORT = 39539
# OSC bundles use the immediate timetag. Keep LAN datagrams below the MTU.
BUNDLE = b'#bundle\0\0\0\0\0\0\0\0\x01'
LAN_PAYLOAD = 1200
INT = Struct('>i')
FLOAT = Struct('>f')
TRANSFORM = Struct('>7f')
UNITY = np.diag([-1., 1., 1.])
PARENT_INDICES = np.asarray(PARENTS[1:])

HUMANOID_NAMES = (
    'Hips', 'LeftUpperLeg', 'RightUpperLeg', 'Spine', 'LeftLowerLeg', 'RightLowerLeg', 'Chest',
    'LeftFoot', 'RightFoot', 'UpperChest', 'LeftToes', 'RightToes', 'Neck', 'LeftShoulder',
    'RightShoulder', 'Head', 'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm', 'RightLowerArm',
    'LeftHand', 'RightHand',
) + tuple(f'{side}{"Little" if finger == "Pinky" else finger}{segment}'
          for side in ('Left', 'Right') for finger in FINGERS
          for segment in ('Proximal', 'Intermediate', 'Distal'))


def osc_string(value):
    data = value.encode('ascii')
    return data + bytes(4 - len(data) % 4)


BONE_HEADER = osc_string('/VMC/Ext/Bone/Pos') + osc_string(',sfffffff')
BONE_PREFIXES = tuple(BONE_HEADER + osc_string(name) for name in HUMANOID_NAMES)
ROOT = (osc_string('/VMC/Ext/Root/Pos') + osc_string(',sfffffff') + osc_string('root')
        + TRANSFORM.pack(0., 0., 0., 0., 0., 0., 1.))
STATUS = osc_string('/VMC/Ext/OK') + osc_string(',i')
TIME = osc_string('/VMC/Ext/T') + osc_string(',f')


def bundles(messages, limit):
    current = bytearray(BUNDLE)
    for message in messages:
        if len(current) + len(message) + INT.size > limit:
            yield bytes(current)
            current = bytearray(BUNDLE)
        current.extend(INT.pack(len(message)))
        current.extend(message)
    yield bytes(current)


class VmcPose:
    """Recover rest-relative world frames and express them as Unity bone locals."""

    def __init__(self, rig, *, normalize_arms=False):
        if (tuple(bone[0] for bone in rig.bones) != NAMES
                or tuple(bone[1] for bone in rig.bones) != PARENTS):
            raise ValueError('VMC needs the Live 52-bone humanoid rig.')
        # The common StreamRig stores rest matrices in Blender Z-up axes.
        rest = Y_TO_Z.T @ np.asarray([bone[2] for bone in rig.bones]).reshape(len(NAMES), 3, 4)
        reference = rest[:, :, :3].copy()
        offsets = rest[:, :, 3].copy()
        offsets[1:] -= offsets[PARENT_INDICES]
        if normalize_arms:
            # MHR's zero pose has lowered arms. Define a T-pose using its
            # own shoulder, elbow and wrist anatomy for normalized VRM rigs.
            t_from_a = np.tile(np.eye(3), (len(NAMES), 1, 1))
            for upper, lower, hand, fingers, sign in ((16, 18, 20, 22, 1.), (17, 19, 21, 37, -1.)):
                direction = np.array([sign, 0., 0.])
                t_from_a[upper] = align(offsets[lower], direction)
                t_from_a[lower] = align(offsets[hand], direction)
                t_from_a[hand] = t_from_a[lower]
                t_from_a[fingers:fingers + 15] = t_from_a[hand]
            reference = t_from_a @ reference
            offsets[1:] = (t_from_a[PARENT_INDICES] @ offsets[1:, :, None])[:, :, 0]
        self.rest_axes_inverse = reference.swapaxes(-1, -2)
        self.offsets = offsets @ UNITY

    def messages(self, world):
        world = np.asarray(world)
        if world.shape != (len(NAMES), 3, 4) or not np.isfinite(world).all():
            raise ValueError('Invalid live pose for VMC.')
        # Rest basis carries the display bone's direction and anatomical roll.
        # Removing it yields identity for every rotation in the reference pose.
        rotations = UNITY @ (world[:, :, :3] @ self.rest_axes_inverse) @ UNITY
        local = rotations.copy()
        local[1:] = rotations[PARENT_INDICES].swapaxes(-1, -2) @ rotations[1:]
        quaternions = Rotation.from_matrix(local).as_quat()
        offsets = self.offsets.copy()
        # VMC's model root stays at identity. All actor travel is on Hips.
        offsets[0] = UNITY @ world[0, :, 3]
        return tuple(prefix + TRANSFORM.pack(*offset, *quaternion)
                     for prefix, offset, quaternion in zip(BONE_PREFIXES, offsets, quaternions))


class VmcStream:
    def __init__(self, host, port):
        try:
            address = IPv4Address(host.strip())
        except ValueError as error:
            raise ValueError('Enter a destination IPv4 address for VMC.') from error
        if address.is_unspecified or address.is_multicast or int(address) == 0xffffffff:
            raise ValueError('Enter a unicast IPv4 address for the VMC receiver.')
        if not 1 <= port <= 65535:
            raise ValueError('VMC port must be between 1 and 65535.')
        self.destination = (str(address), port)
        # Loopback can carry a complete pose atomically. Split frames on LANs
        # so standard-size Ethernet packets do not need IP fragmentation.
        self.limit = 8192 if address.is_loopback else LAN_PAYLOAD
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.setblocking(False)
        except OSError:
            self.socket.close()
            raise
        self.started_at = monotonic()
        self.last_sent = float('-inf')
        self.rig_key = None
        self.pose = None
        self.latest = ()
        self.problem = None

    def _send(self):
        if self.socket is None:
            return
        now = monotonic()
        frame = (ROOT, *self.latest) if self.latest else ()
        stamp = TIME + FLOAT.pack(now - self.started_at)
        status = STATUS + INT.pack(int(bool(self.latest)))
        try:
            for packet in bundles((*frame, stamp, status), self.limit):
                self.socket.sendto(packet, self.destination)
        except BlockingIOError:
            # Drop this datagram rather than queue it or block the GUI.
            pass
        except OSError as error:
            self.problem = f'VMC UDP send failed for {self.destination[0]}:{self.destination[1]}. {error}'
        else:
            self.problem = None
        self.last_sent = now

    def publish(self, actors, selected, *, normalize_arms=False):
        if not actors:
            self.latest = ()
        else:
            _, _, rig, _, world, transform = actors[0]
            if transform is not None:
                raise ValueError('VMC expects Live world coordinates.')
            key = rig.key, normalize_arms
            if key != self.rig_key:
                self.pose = VmcPose(rig, normalize_arms=normalize_arms)
                self.rig_key = key
            self.latest = self.pose.messages(world)
        self._send()

    def refresh(self):
        # Refresh a paused pose for receivers with a timeout and announce
        # unavailability periodically while waiting for a valid tracking pose.
        if monotonic() - self.last_sent >= (.15 if self.latest else 1.):
            self._send()

    def error(self):
        return self.problem

    def close(self):
        if self.socket is not None:
            self.latest = ()
            self._send()
            self.socket.close()
            self.socket = None
