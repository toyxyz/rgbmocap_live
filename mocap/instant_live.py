"""InstantHMR CUDA inference and a bone-only MHR adapter for the Live workspace."""
from hashlib import sha256 as bytes_sha256
from io import BytesIO
import math
import os
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .bone_overlay import reference_frame
from .instant_alignment import InstantCamera, assumed_focal, assumed_intrinsics, project
from .live_pose import PARENTS
from .models import BY_KEY
from .paths import ROOT
from .pear.bone_roll import JOINTS, TIPS, SkeletonLayout
from .pear.camera_pose import CV_AXES
from .pear.multiview import BODY_IDS, fk
from .pear.temporal_filter import AdaptiveFilter, strength_scale
from .storage import sha256

# Joint IDs checked against Meta MHR v1.2.0 rig.npz and the pinned body graph.
# Collapse the extra spine root, metacarpals, ankle pivots and wrist twist
# through the native global transforms before deriving the Live local channels.
BODY = (1, 2, 18, 34, 3, 19, 35, 4, 20, 37, 8, 24, 110, 74, 38, 113, 75, 39, 76, 40, 78, 42)
FINGERS = (92, 93, 94, 88, 89, 90, 80, 81, 82, 84, 85, 86, 97, 98, 99,
           56, 57, 58, 52, 53, 54, 44, 45, 46, 48, 49, 50, 61, 62, 63)
MHR_JOINTS = np.array(BODY + FINGERS)
MHR_ENDS = {15: 126, 27: 95, 30: 91, 33: 83, 36: 87, 39: 100,
            42: 59, 45: 55, 48: 47, 51: 51, 54: 64}
NATIVE_PARENTS = np.concatenate((PARENTS[:22], [15] * 3,
    [JOINTS[i] for i in PARENTS[22:]])).astype(np.int32)
PARENT = np.asarray(PARENTS[1:])
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def model_path(key):
    model = BY_KEY[key]
    if not model.path.is_file() or sha256(model.path) != model.expected_sha256:
        raise ValueError(f'Prepare the official {model.name} file in Model Manager.')
    return model.path


def skeleton_graph(path):
    """Keep only the pinned MHR graph's global skeleton state, before skinning."""
    import onnx
    from onnx import TensorProto, helper, utils

    model = onnx.load(str(path), load_external_data=False)
    state = '/Where_3_output_0'
    model.graph.value_info.append(helper.make_tensor_value_info(state, TensorProto.FLOAT, ['batch', 127, 8]))
    graph = utils.Extractor(model).extract_model(['mhr_params'], [state])
    return graph.SerializeToString()


def network_input(bgr, box, size, use_focal):
    height, width = bgr.shape[:2]
    center = (box[:2] + box[2:]) * .5
    span = float(max(box[2:] - box[:2]))
    if span < 2.:
        raise ValueError('The detected person is too small.')
    focal = assumed_focal(width, height)
    dx, dy = center - (width * .5, height * .5)
    cliff = ([math.atan(dx / focal), math.atan(dy / focal), span / focal] if use_focal
             else [2 * dx / width, 2 * dy / height, span / max(width, height)])
    lower = np.floor(center - .6 * span)
    upper = np.ceil(center + .6 * span)
    sx, sy = (upper - lower) / size
    transform = np.array([[sx, 0, lower[0] + .5 * sx - .5], [0, sy, lower[1] + .5 * sy - .5]])
    crop = cv2.warpAffine(bgr, transform, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    image = (crop[:, :, ::-1].astype(np.float32) / 255. - MEAN) / STD
    return dict(image=np.ascontiguousarray(image.transpose(2, 0, 1)[None]),
                cliff_cond=np.array([cliff], np.float32))


class InstantRig:
    revision = 1

    def stream_reference(self):
        return self.reference

    def __init__(self, neutral):
        self.layout = SkeletonLayout(NATIVE_PARENTS)
        self.inverse = Rotation.from_quat(neutral[MHR_JOINTS, 3:7]).as_matrix().swapaxes(-2, -1)
        rest = neutral[MHR_JOINTS, :3] * .01
        # Supported feet point in +Z. Approximate the sole under the heel from
        # the neutral toe joint, which is the lowest skeletal pivot.
        origin = rest[0].copy()
        origin[1] = min(rest[10, 1], rest[11, 1]) - .02
        self.rest = rest - origin
        self.offsets = self.rest[1:] - self.rest[PARENT]
        self.tips = {}
        neutral_joints = np.zeros((145, 3), dtype=np.float32)
        neutral_joints[JOINTS] = self.rest
        for i, native_joint in enumerate(JOINTS):
            if native_joint not in TIPS:
                continue
            if native_joint in MHR_ENDS:
                offset = neutral[MHR_ENDS[native_joint], :3] * .01 - rest[i]
            else:
                offset = np.array([0., 0., .10], np.float32)
            tip = TIPS[native_joint]
            self.tips[tip] = (i, offset)
            neutral_joints[tip] = self.rest[i] + offset
        heads, ends = self.layout.landmarks(neutral_joints)
        directions = self.layout.directions(heads, ends)
        self.basis = np.array([reference_frame(vector) for vector in directions], np.float32)
        self.reference = dict(rest_heads=np.ascontiguousarray(heads, dtype=np.float32),
                              rest_basis=self.basis, rest_lengths=np.linalg.norm(directions, axis=1).astype(np.float32),
                              rest_parents=np.asarray(PARENTS, np.int32))
        native_rest = np.zeros((55, 3), dtype=np.float32)
        native_rest[JOINTS] = self.rest
        native_rest[22:25] = native_rest[15]
        self.fusion_rig = SimpleNamespace(
            native_parents=NATIVE_PARENTS, rest=native_rest, basis=self.basis,
            tip_offsets={int(JOINTS[i]): offset for i, offset in self.tips.values()})
        self.rotation_filter = AdaptiveFilter(rotation=True)
        self.position_filter = AdaptiveFilter(beta=4., vector=True)
        self.reset()

    def reset_filter(self):
        self.rotation_filter.reset()
        self.position_filter.reset()
        self.last_time = None

    def reset(self):
        self.reset_filter()
        self.origin = None

    def local_rotations(self, state):
        source = Rotation.from_quat(state[MHR_JOINTS, 3:7]).as_matrix() @ self.inverse
        local = source.copy()
        local[1:] = source[PARENT].swapaxes(-2, -1) @ source[1:]
        return local

    def fusion_view(self, packet, rig, world_from_camera):
        local = np.broadcast_to(np.eye(3), (55, 3, 3)).copy()
        local[JOINTS] = self.local_rotations(packet['state'])
        local[0] = world_from_camera @ CV_AXES @ local[0]
        joints, _ = fk(local, rig)
        return local, packet['uv'], joints

    def pose(self, state, camera, timestamp, rotations, position, strength, *, alignment=None):
        local = self.local_rotations(state)
        root = state[MHR_JOINTS[0], :3] * .01
        if alignment is not None:
            local[0] = alignment.pose.basis @ local[0]
            body = state[MHR_JOINTS[BODY_IDS], :3] * .01 @ CV_AXES
            camera = alignment.translation(body, camera)
            root = alignment.pose.center + alignment.pose.basis @ (root + CV_AXES @ camera)
        else:
            root = root + CV_AXES @ camera
        dt = 0. if self.last_time is None else timestamp - self.last_time
        self.last_time = timestamp
        scale = strength_scale(strength)
        for filter_, beta in ((self.rotation_filter, .65), (self.position_filter, 4.)):
            filter_.min_cutoff, filter_.beta, filter_.max_cutoff = 1.2 / scale, beta / scale, 8. / scale
        if rotations:
            local = self.rotation_filter.update(local, dt)
        else:
            self.rotation_filter.reset()
        if position:
            root = self.position_filter.update(root, dt)
        else:
            self.position_filter.reset()
        world = np.empty((52, 3, 3), np.float32)
        points = np.empty((52, 3), np.float32)
        world[0], points[0] = local[0], root
        for i, parent in enumerate(PARENTS[1:], 1):
            world[i] = world[parent] @ local[i]
            points[i] = points[parent] + world[parent] @ self.offsets[i - 1]
        if self.origin is None and alignment is None:
            self.origin = points[0].copy()
            self.origin[1] = min(points[10, 1], points[11, 1]) - .02
        joints = np.zeros((145, 3), np.float32)
        joints[JOINTS] = points if alignment is not None else points - self.origin
        for tip, (i, offset) in self.tips.items():
            joints[tip] = joints[JOINTS[i]] + world[i] @ offset
        native = np.empty((55, 3, 3), np.float32)
        native[JOINTS] = world
        native[22:25] = world[15]
        return dict(bounds=joints[np.r_[JOINTS, list(self.tips)]], joints=joints, rotations=native,
                    bone_facing=world @ self.basis, **self.reference)


def load_detector():
    """Allow the pickle loader only for the exact existing PEAR detector bytes."""
    import torch
    settings = ROOT / 'cache/instanthmr/yolo'
    settings.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings)
    from ultralytics import YOLO
    from ultralytics.nn import tasks

    spec = BY_KEY['pear_yolo']
    path = spec.path
    payload = path.read_bytes()
    if bytes_sha256(payload).hexdigest() != spec.expected_sha256:
        raise ValueError('Prepare the official person detector in Model Manager.')
    default_loader = tasks.torch_safe_load

    def load_verified(weight, *args, **kwargs):
        if Path(weight).resolve() != path.resolve():
            return default_loader(weight, *args, **kwargs)
        return torch.load(BytesIO(payload), map_location='cpu', weights_only=False), str(path)

    tasks.torch_safe_load = load_verified
    try:
        return YOLO(str(path))
    finally:
        tasks.torch_safe_load = default_loader


class InstantHMR:
    parents = NATIVE_PARENTS

    def __init__(self, progress):
        start = perf_counter()
        from .instant_runtime import activate
        activate()
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 2
        progress('Loading InstantHMR CUDA inference…')
        self.network = ort.InferenceSession(str(model_path('instant_network')), options,
            providers=[('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC'}), 'CPUExecutionProvider'])
        if self.network.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('InstantHMR CUDA could not start. Run Setup_InstantHMR.cmd.')
        self.network.disable_fallback()
        meta = self.network.get_modelmeta().custom_metadata_map
        self.size = int(meta['image_size'])
        self.use_focal = meta['cliff_focal'] == 'true'
        # This small skeleton graph is CPU-resident. No mesh is evaluated
        # and no mesh-sized data is moved across the GPU or worker boundary.
        progress('Loading the MHR skeleton…')
        self.body = ort.InferenceSession(skeleton_graph(model_path('instant_body')),
                                         options, providers=['CPUExecutionProvider'])
        self.rig = InstantRig(self.state(np.zeros((1, 204), np.float32)))
        progress('Loading the person detector…')
        self.detector = load_detector()
        self.previous = None
        self.alignment = None
        self.alignment_key = None
        self.load_seconds = perf_counter() - start

    def state(self, params):
        return self.body.run(None, {'mhr_params': params})[0][0]

    def warmup(self):
        image = np.zeros((480, 640, 3), np.uint8)
        self.predict(image, np.array([160, 10, 480, 470], dtype=np.float32))
        self.detect(image)

    def detect_batch(self, images):
        predictions = self.detector.predict(images, device=0, classes=[0], conf=.5, imgsz=640, verbose=False)
        return [prediction.boxes.xyxy.detach().cpu().numpy() for prediction in predictions]

    def detect(self, image):
        return self.detect_batch([image])[0]

    def predict(self, image, box):
        params, camera = self.network.run(['mhr_params', 'cam_trans'],
                                          network_input(image, box, self.size, self.use_focal))
        return self.state(params), camera[0]

    def predict_batch(self, crops):
        inputs = {key: np.concatenate([crop[key] for crop in crops]) for key in crops[0]}
        params, translations = self.network.run(['mhr_params', 'cam_trans'], inputs)
        states = self.body.run(None, {'mhr_params': params})[0]
        return zip(states, translations)

    def configure_camera(self, request, shape):
        height, width = shape[:2]
        folder, edits = request.get('calibration'), request.get('camera_edits')
        signature = request.get('signature')
        key = folder, edits, signature, width, height
        if self.alignment_key != key:
            self.alignment = InstantCamera.load(folder, edits, signature, width, height)
            self.alignment_key = key
            self.rig.reset()

    def alignment_observation(self, image):
        box = select_person(self.detect(image), None)
        if box is None:
            raise ValueError('Camera Align needs a visible person in both samples.')
        packet = view_packet(image, box, *self.predict(image, box))
        if packet is None:
            raise ValueError('Camera Align needs valid body depth in both samples.')
        return packet['uv'], packet['bbox'], packet['camera_relative']

    def run(self, image, timestamp, rotations, position, strength):
        self.previous = select_person(self.detect(image), self.previous)
        if self.previous is None:
            self.rig.reset_filter()
            return None
        state, camera = self.predict(image, self.previous)
        if not np.isfinite(state).all() or not np.isfinite(camera).all() or camera[2] <= 0.:
            self.rig.reset_filter()
            return None
        return self.rig.pose(state, camera, timestamp, rotations, position, strength, alignment=self.alignment)


def select_person(boxes, previous):
    if not len(boxes):
        return None
    sizes = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    if previous is None:
        return boxes[sizes.argmax()].copy()
    overlap = np.maximum(0., np.minimum(boxes[:, 2:], previous[2:]) -
                         np.maximum(boxes[:, :2], previous[:2])).prod(axis=1)
    iou = overlap / np.maximum(1., sizes + (previous[2:] - previous[:2]).prod() - overlap)
    best = iou.argmax()
    return boxes[best].copy() if iou[best] >= .1 else None


def view_packet(image, box, state, camera):
    if not np.isfinite(state).all() or not np.isfinite(camera).all():
        return None
    relative = np.zeros((55, 3))
    relative[JOINTS] = state[MHR_JOINTS, :3] * .01 @ CV_AXES
    relative[22:25] = relative[15]
    camera_joints = relative + camera
    if np.any(camera_joints[BODY_IDS, 2] <= .01):
        return None
    height, width = image.shape[:2]
    uv = np.full((145, 2), np.nan)
    visible = np.flatnonzero(camera_joints[:, 2] > .01)
    uv[visible] = project(camera_joints[visible], assumed_intrinsics(width, height))
    # Ignore invalid views rather than allowing nonfinite weights in fusion.
    if not np.isfinite(uv[:55]).all():
        return None
    return dict(state=state, camera=camera, uv=uv, camera_relative=relative,
                bbox=box, image_size=np.array([width, height]))


class InstantBatch:
    def __init__(self, model, pool):
        self.model = model
        self.pool = pool
        self.previous = []

    def reset(self, count):
        self.previous = [None] * count

    def run(self, images):
        if len(self.previous) != len(images):
            self.reset(len(images))
        packets = [None] * len(images)
        for first in range(0, len(images), 4):
            group = images[first:first + 4]
            indexes = []
            for index, detections in enumerate(self.model.detect_batch(group), first):
                box = select_person(detections, self.previous[index])
                self.previous[index] = box
                if box is not None and max(box[2:] - box[:2]) >= 2.:
                    indexes.append(index)
            if indexes:
                crops = list(self.pool.map(lambda i: network_input(
                    images[i], self.previous[i], self.model.size, self.model.use_focal), indexes))
                for index, (state, camera) in zip(indexes, self.model.predict_batch(crops)):
                    packets[index] = view_packet(images[index], self.previous[index], state, camera)
        return packets

    def close(self):
        self.reset(0)
