"""Selectable input cameras and a combined transform gizmo for RGB MoCap Live."""
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen

from .pear.camera_pose import CameraPose
from .viewport import MotionView


AXES = np.eye(3)
AXIS_COLORS = ('#ff6666', '#87df55', '#62aaff')
CAMERA_EDGES = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
RING_ANGLES = np.linspace(0., 2. * math.pi, 49)


def point(xy):
    return QPointF(*xy)


def line_distance(cursor, start, end):
    segment = end - start
    length = segment @ segment
    fraction = np.clip((cursor - start) @ segment / max(length, 1e-9), 0., 1.)
    return np.linalg.norm(cursor - start - fraction * segment)


class LiveCameraViewport(MotionView):
    camera_selected = Signal(int)
    camera_edited = Signal(int, object)

    def __init__(self):
        super().__init__()
        self.show_cameras = True
        self.input_cameras = []
        self.selected_camera = -1
        self._camera_hits = []
        self._transform_hits = []
        self._transform_drag = None
        self.camera_editing = True
        self.setToolTip('Click an input camera. Drag arrows to move or rings to rotate around world axes.')

    def set_input_cameras(self, cameras):
        self.input_cameras = cameras
        self._camera_hits = []
        self._transform_hits = []
        if self.selected_camera >= len(cameras):
            self.select_camera(-1)
        self.update()

    def set_camera_editing(self, enabled):
        self.camera_editing = enabled
        if not enabled and self._transform_drag is not None:
            self._end_transform()

    def select_camera(self, index):
        if index != self.selected_camera:
            self._transform_drag = None
            self._transform_hits = []
            self.selected_camera = index
            self.camera_selected.emit(index)
            self.update()

    def target(self):
        if self._transform_drag is not None:
            return self._transform_drag['target']
        return super().target()

    def frame_all(self):
        if not (self.live_active and self.show_cameras and self.input_cameras):
            return super().frame_all()
        points = [pose.center for pose, _, _ in self.input_cameras]
        if self.live_data is not None:
            points.extend(self.live_data['bounds'])
        else:
            points.extend(([0., 0., 0.], [0., 1.8, 0.]))
        points = np.asarray(points)
        center = (points.min(0) + points.max(0)) * .5
        self.scene_center = center.copy()
        self.pan[:] = 0
        self.pan = center - self.target()
        radius = np.linalg.norm(points - center, axis=1).max() + .45
        aspect = max(.35, min(1., self.width() / max(1, self.height())))
        self.distance = radius / math.sin(math.atan(aspect * math.tan(math.radians(20))))
        self.auto_fit = False
        self.update()

    def paintGL(self):
        super().paintGL()
        if self.live_active:
            self._paint_input_cameras()

    def paint_scene(self, event):
        super().paint_scene(event)
        self._paint_input_cameras()

    def _paint_input_cameras(self):
        self._camera_hits = []
        self._transform_hits = []
        if not (self.live_active and self.show_cameras and self.input_cameras):
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            for index, (pose, state, view_shape) in enumerate(self.input_cameras):
                half_width, half_height = view_shape
                corners = np.array([[0., 0., 0.], [-half_width, -half_height, -.42],
                    [half_width, -half_height, -.42], [half_width, half_height, -.42],
                    [-half_width, half_height, -.42]])
                screen, depth, _ = self.project(pose.center + corners @ pose.basis.T)
                if depth[0] <= .08:
                    continue
                color = QColor('#8edbff' if index == self.selected_camera
                               else '#ffc479' if state != 'Estimated' else '#bbc6cf')
                painter.setPen(QPen(color, 2))
                lines = [(screen[a], screen[b]) for a, b in CAMERA_EDGES if min(depth[a], depth[b]) > .08]
                for start, end in lines:
                    painter.drawLine(point(start), point(end))
                painter.setBrush(color)
                painter.drawEllipse(point(screen[0]), 4, 4)
                label = f'Camera {index + 1}'
                fm = painter.fontMetrics()
                area = QRectF(screen[0, 0] + 9, screen[0, 1] + 5,
                              fm.horizontalAdvance(label) + 12, fm.height() + 4)
                painter.setBrush(QColor('#202932'))
                painter.drawRoundedRect(area, 4, 4)
                painter.drawText(area, Qt.AlignCenter, label)
                self._camera_hits.append((index, area, screen[0], lines))
            if self.selected_camera >= 0 and self.camera_editing:
                self._draw_transform(painter, self.input_cameras[self.selected_camera][0])
        finally:
            painter.end()

    def _draw_transform(self, painter, pose):
        origin, depth, focal = self.project(pose.center)
        if depth <= .08:
            return
        scale = 87. * depth / focal
        for index in range(3):
            color = QColor(AXIS_COLORS[index])
            u, v = AXES[(index + 1) % 3], AXES[(index + 2) % 3]
            circle = .64 * scale * (np.cos(RING_ANGLES)[:, None] * u
                                    + np.sin(RING_ANGLES)[:, None] * v)
            ring, depths, _ = self.project(pose.center + circle)
            color.setAlpha(170)
            painter.setPen(QPen(color, 2))
            for i in range(len(ring) - 1):
                start, end = ring[i:i+2]
                if min(depths[i:i+2]) > .08:
                    painter.drawLine(point(start), point(end))
                    tangent = (end - start) / (RING_ANGLES[i + 1] - RING_ANGLES[i])
                    if np.linalg.norm(tangent) > 8:
                        self._transform_hits.append(('rotate', index, start, end, tangent))

        for index, axis in enumerate(AXES):
            ends, depths, _ = self.project(pose.center + np.array([.19, 1.])[:, None] * scale * axis)
            start, end = ends
            direction = end - start
            length = np.linalg.norm(direction)
            if min(depths) <= .08 or length < 12:
                continue
            unit = direction / length
            wing = np.array([-unit[1], unit[0]]) * 5
            painter.setPen(QPen(QColor(AXIS_COLORS[index]), 3))
            painter.drawLine(point(start), point(end))
            for tip in (end - unit * 11 + wing, end - unit * 11 - wing):
                painter.drawLine(point(tip), point(end))
            # Translation uses the screen derivative at the camera origin.
            tangent = (self.project(pose.center + axis * scale * .02)[0] - origin) / (scale * .02)
            self._transform_hits.append(('move', index, start, end, tangent))

    def _transform_at(self, cursor):
        if not self.camera_editing:
            return None
        hits = [(line_distance(cursor, item[2], item[3]), item) for item in self._transform_hits]
        distance, hit = min(hits, default=(float('inf'), None), key=lambda entry: entry[0])
        return hit if distance <= 8 else None

    def _input_camera_at(self, cursor):
        for index, area, _, _ in reversed(self._camera_hits):
            if area.contains(point(cursor)):
                return index
        hits = [(min([np.linalg.norm(cursor - center)]
                     + [line_distance(cursor, a, b) for a, b in lines]), index)
                for index, _, center, lines in self._camera_hits]
        distance, index = min(hits, default=(float('inf'), -1))
        return index if distance <= 9 else -1

    def mousePressEvent(self, event):
        if self.live_active and self.show_cameras and event.button() == Qt.LeftButton:
            cursor = np.array([event.position().x(), event.position().y()])
            for other, area, _, _ in reversed(self._camera_hits):
                if other != self.selected_camera and area.contains(point(cursor)):
                    self.select_camera(other)
                    self.drag_point = None
                    event.accept()
                    return
            hit = self._transform_at(cursor)
            if hit is not None:
                kind, index, _, _, tangent = hit
                pose, state, _ = self.input_cameras[self.selected_camera]
                self._transform_drag = dict(kind=kind, index=index, initial=pose,
                    edited=state == 'Manual', cursor=cursor, tangent=tangent,
                    target=self.target(), observer=self.camera(), changed=False)
                self.drag_point = None
                self.setFocus(Qt.MouseFocusReason)
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
            index = self._input_camera_at(cursor)
            if index >= 0:
                self.select_camera(index)
                self.drag_point = None
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        cursor = np.array([event.position().x(), event.position().y()])
        drag = self._transform_drag
        if drag is None:
            if self.show_cameras and self.drag_point is None:
                if self._transform_at(cursor) is not None or self._input_camera_at(cursor) >= 0:
                    self.setCursor(Qt.PointingHandCursor)
                    return
            return super().mouseMoveEvent(event)
        if not (event.buttons() & Qt.LeftButton):
            self._end_transform()
            return
        delta = cursor - drag['cursor']
        tangent = drag['tangent']
        amount = (delta @ tangent) / max(tangent @ tangent, 1e-9)
        initial = drag['initial']
        axis = AXES[drag['index']]
        if drag['kind'] == 'move':
            pose = CameraPose(initial.center + amount * axis, initial.basis)
        else:
            start_angle = self._ring_angle(drag, drag['cursor'])
            end_angle = self._ring_angle(drag, cursor)
            if start_angle is not None and end_angle is not None:
                amount = end_angle - start_angle
            x, y, z = axis
            cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            rotation = np.eye(3) + math.sin(amount) * cross + (1. - math.cos(amount)) * (cross @ cross)
            pose = CameraPose(initial.center, rotation @ initial.basis)
        if np.linalg.norm(delta) >= 2 or drag['changed']:
            drag['changed'] = True
            self.camera_edited.emit(self.selected_camera, pose)
        event.accept()

    def _ring_angle(self, drag, screen):
        eye, axes = drag['observer']
        pose = drag['initial']
        index = drag['index']
        focal = self.height() * .5 / math.tan(math.radians(20))
        ray = axes[2] + axes[0] * (screen[0] - self.width() * .5) / focal
        ray -= axes[1] * (screen[1] - self.height() * .5) / focal
        if abs(ray[index]) < .15:
            return None
        depth = (pose.center[index] - eye[index]) / ray[index]
        local = eye + depth * ray - pose.center
        u, v = local[(index + 1) % 3], local[(index + 2) % 3]
        if depth <= 0 or math.hypot(u, v) < .035:
            return None
        return math.atan2(v, u)

    def _end_transform(self):
        self._transform_drag = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._transform_drag is not None and event.button() == Qt.LeftButton:
            self._end_transform()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        drag = self._transform_drag
        if drag is not None and event.key() == Qt.Key_Escape:
            if drag['changed']:
                self.camera_edited.emit(self.selected_camera, drag['initial'] if drag['edited'] else None)
            self._end_transform()
            event.accept()
        else:
            super().keyPressEvent(event)
