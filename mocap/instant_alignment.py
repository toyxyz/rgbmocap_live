"""Single-view InstantHMR observations and fixed-camera placement."""
from dataclasses import dataclass
import json
import math
from time import monotonic

import cv2
import numpy as np

from .pear.bone_roll import JOINTS
from .pear.camera_pose import CameraPose
from .pear.input_transform import rotate_frame
from .pear.multi_worker import create_sample_folder, owned_folder, write_sample
from .pear.multiview import BODY_IDS
from .pear.single_alignment import fit_translation
from .storage import atomic_json, sha256


def assumed_focal(width, height):
    return 1.05 * math.hypot(width, height)


def assumed_intrinsics(width, height):
    focal = assumed_focal(width, height)
    return np.array([[focal, 0., width * .5], [0., focal, height * .5], [0., 0., 1.]])


def project(joints, intrinsics):
    homogeneous = joints @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


@dataclass
class InstantCamera:
    pose: CameraPose
    intrinsics: np.ndarray
    estimated_intrinsics: np.ndarray
    solved: bool

    @classmethod
    def load(cls, folder, edits, signature, width, height):
        base = assumed_intrinsics(width, height)
        intrinsics = base
        extrinsic = None
        if folder:
            data = json.loads((owned_folder(folder) / 'cameras.json').read_text('utf8'))
            k, e = np.asarray(data['intrinsics'], dtype=float), np.asarray(data['extrinsics'], dtype=float)
            if (data['signature'] != signature or data['sizes'] != [[width, height]]
                    or k.shape != (1, 3, 3) or e.shape != (1, 3, 4)
                    or not np.isfinite(k).all() or (k[0, [0, 1], [0, 1]] <= 0).any()
                    or not np.allclose(k[0, 2], [0., 0., 1.])):
                raise ValueError('Camera alignment does not match this input. Run Camera Align again.')
            intrinsics, extrinsic = k[0], e[0]
        if edits:
            if len(edits) != 1:
                raise ValueError('InstantHMR supports one manually placed camera.')
            if edits[0] is not None:
                extrinsic = np.asarray(edits[0], dtype=float)
        if extrinsic is None:
            return None
        if (extrinsic.shape != (3, 4) or not np.isfinite(extrinsic).all()
                or not np.allclose(extrinsic[:, :3].T @ extrinsic[:, :3], np.eye(3), atol=1e-4)
                or np.linalg.det(extrinsic[:, :3]) < .999):
            raise ValueError('Invalid camera rotation or position.')
        return cls(CameraPose.from_extrinsic(extrinsic), intrinsics, base, bool(folder))

    def translation(self, joints, estimate):
        if not self.solved:
            return estimate
        camera_joints = joints + estimate
        if np.any(camera_joints[:, 2] <= .01):
            return estimate
        uv = project(camera_joints, self.estimated_intrinsics)
        result = fit_translation(joints, uv, self.intrinsics)
        return result if result is not None else estimate


def prepare_alignment(model, request, angle, camera, cap):
    """Take two rotated samples with the existing capture or video reader."""
    if camera is None:
        if cap is None:
            raise ValueError('Open a video or assign a webcam before Camera Align.')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        timeline_fps = float(request['fps'])
        start = int(request['frame'])
        end = round(float(request['duration']) * timeline_fps) - 1
        if fps <= 0. or not math.isfinite(fps) or end <= start:
            raise ValueError('Choose a playhead with at least one later frame for Camera Align.')
        marks = (start, min(start + round(2. * timeline_fps), end))
    else:
        marks = (None, None)

    samples = []
    for mark in marks:
        if camera is not None:
            deadline = monotonic() + 2.
            item = None
            while item is None and monotonic() < deadline:
                item = camera.take()
            if item is None:
                raise ValueError('Webcam did not deliver fresh samples for Camera Align.')
            image = item[0]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(mark * fps / timeline_fps))
            ok, image = cap.read()
            if not ok:
                raise ValueError('Cannot decode the selected video frames for Camera Align.')
        image = rotate_frame(image, angle)
        height, width = image.shape[:2]
        if samples and (width, height) != samples[0][0]:
            raise ValueError('Camera dimensions changed during alignment.')
        uv, box, relative = model.alignment_observation(image)
        scale = min(1., 1920 / width, 1080 / height)
        if scale < 1.:
            image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
        samples.append(((width, height), image, uv, box, relative))

    folder = create_sample_folder(request.get('calibration'))
    paths = []
    for i, (_, image, _, _, _) in enumerate(samples):
        path = folder / f'sample-{i}-0.png'
        write_sample(path, image, cv2)
        paths.append(path.name)
    rest = np.zeros((55, 3), np.float32)
    rest[JOINTS] = model.rig.rest
    np.savez(folder / 'rig.npz', rest=rest)
    np.savez(folder / 'observations.npz',
             uv=np.stack([sample[2] for sample in samples])[:, None],
             boxes=np.stack([sample[3] for sample in samples])[:, None],
             sizes=np.asarray([samples[0][0]]), rest=rest,
             camera_relative=np.stack([sample[4] for sample in samples])[:, None])
    atomic_json(folder / 'job.json', dict(version=1, source_model='instant', signature=request['signature'], images=paths,
                count=1, rig_sha256=sha256(folder / 'rig.npz'),
                observations_sha256=sha256(folder / 'observations.npz')))
    return folder
