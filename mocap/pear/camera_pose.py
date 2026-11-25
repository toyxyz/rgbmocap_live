"""Fixed-camera poses shared by the PEAR viewport and fusion worker."""
from dataclasses import dataclass
import math

import numpy as np


DISPLAY = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])
CV_AXES = np.diag([1., -1., -1.])


@dataclass(frozen=True)
class CameraPose:
    center: np.ndarray
    basis: np.ndarray  # Viewport camera local right, up, and back axes in world space.

    @classmethod
    def from_extrinsic(cls, extrinsic):
        rotation = extrinsic[:, :3]
        center = -DISPLAY @ rotation.T @ extrinsic[:, 3]
        return cls(center, DISPLAY @ rotation.T @ CV_AXES)

    def extrinsic(self):
        rotation = CV_AXES @ self.basis.T @ DISPLAY
        return np.column_stack((rotation, -rotation @ DISPLAY.T @ self.center))


def assumed_camera(index):
    """Editable starting pose for an input without background calibration."""
    step = (index + 1) // 2
    angle = (1 if index % 2 else -1) * math.radians(min(75, 18 * step))
    radius = 3.2 + max(0, step - 4) * .22
    center = np.array([radius * math.sin(angle), 1.45, -radius * math.cos(angle)])
    forward = np.array([0., 1.45, 0.]) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    return CameraPose(center, np.column_stack((right, np.cross(right, forward), -forward)))


def assumed_intrinsics(width, height):
    focal = 1.2 * max(width, height)
    return [[focal, 0., width * .5], [0., focal, height * .5], [0., 0., 1.]]


def resolve_cameras(automatic, edits, sizes):
    if automatic:
        K = np.asarray(automatic['intrinsics'], dtype=float)
        E = np.asarray(automatic['extrinsics'], dtype=float).copy()
        active = [True] * len(sizes)
    else:
        K = [assumed_intrinsics(*size) for size in sizes]
        E = np.stack([assumed_camera(i).extrinsic() for i in range(len(sizes))])
        active = [False] * len(sizes)

    if edits is not None:
        if not isinstance(edits, list) or len(edits) != len(sizes):
            raise ValueError('Manual cameras do not match the current inputs.')
        for index, extrinsic in enumerate(edits):
            if extrinsic is None:
                continue
            matrix = np.asarray(extrinsic, dtype=float)
            if matrix.shape != (3, 4):
                raise ValueError('Invalid manual camera pose.')
            E[index] = matrix
            active[index] = True
    return dict(intrinsics=K, extrinsics=E), active
