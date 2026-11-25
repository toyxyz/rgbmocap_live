"""Convert PEAR's calibrated rotations and points to the receiver contract."""
from hashlib import sha256

import numpy as np

from .bone_stream import StreamRig
from .mixamo_rig import BODY_NAMES, BODY_PARENTS
from .pear.bone_roll import JOINTS

# Native SMPL-X phalanges are in Index, Middle, Pinky, Ring, Thumb order.
FINGERS = ('Index', 'Middle', 'Pinky', 'Ring', 'Thumb')
NAMES = BODY_NAMES + tuple(f'{side}Hand{finger}{joint}'
                           for side in ('Left', 'Right')
                           for finger in FINGERS for joint in (1, 2, 3))
PARENTS = BODY_PARENTS + tuple(
    (20 + side if digit == 0 else 22 + side * 15 + finger * 3 + digit - 1)
    for side in range(2) for finger in range(5) for digit in range(3)
)
ACTOR = sha256(b'toyxyz.rgbmocap_live.actor.v1').hexdigest()
LABEL = 'Person 1'


class LivePose:
    def __init__(self):
        self.key = None
        self.rig = None
        self.sequence = 0

    def packet(self, source, metadata, pose):
        revision = metadata['bone_revision']
        key = (source, revision)
        if key != self.key:
            heads = pose['rest_heads']
            basis = pose['rest_basis']
            lengths = pose['rest_lengths']
            parents = pose['rest_parents']
            if (heads.shape != (52, 3) or basis.shape != (52, 3, 3)
                    or lengths.shape != (52,) or not np.array_equal(parents, PARENTS)
                    or np.any(lengths <= 0)):
                raise ValueError('PEAR live rig has an incompatible hierarchy or lengths.')
            self.rig = StreamRig(NAMES, parents, heads, basis, np.eye(3, dtype=np.float32), lengths)
            self.key = key
        world = np.empty((52, 3, 4), dtype=np.float32)
        world[:, :, :3] = pose['bone_facing']
        world[:, :, 3] = pose['joints'][JOINTS]
        self.sequence = (self.sequence + 1) % 2147483647
        return (ACTOR, LABEL, self.rig, self.sequence, world, None)
