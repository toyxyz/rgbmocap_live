"""PEAR workspace with an independent live sender and original controls."""
from collections import deque
import json
from pathlib import Path
from time import monotonic

import numpy as np
from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .bone_stream import BoneStream, PORT
from .gui import MainWindow
from .live_camera_view import LiveCameraViewport
from .live_pose import ACTOR, LABEL, LivePose
from .paths import ROOT
from .pear.camera_pose import CameraPose, assumed_camera
from .unreal_stream import UnrealStream
from .vmc_stream import VMC_HOST, VMC_PORT, VmcStream

STALE_SECONDS = .75


class LiveWindow(MainWindow):
    def __init__(self, video=None):
        self._live_backend = 'pear'
        self._live_camera_signature = None
        self._live_camera_baseline = None
        self._live_auto_cameras = []
        self._instant_camera_estimate = None
        self._live_camera_edits = {}
        super().__init__(None)
        self.build_live_camera_controls()
        self.people_tabs.removeTab(0)
        self.people_panel.show()
        if video is not None:
            self.load_pear_videos([video])

    def create_motion_view(self):
        view = LiveCameraViewport()
        view.camera_selected.connect(self.live_camera_selected)
        view.camera_edited.connect(self.live_camera_edited)
        return view

    def build_live_camera_controls(self):
        footer = self.viewport_footer.layout()
        self.live_camera_picker = QComboBox()
        self.live_camera_picker.setAccessibleName('Input camera to edit')
        self.live_camera_picker.setToolTip('Select a camera even if its label overlaps another camera.')
        self.live_camera_picker.currentIndexChanged.connect(
            lambda row: self.motion_view.select_camera(row - 1))
        self.live_camera_reset = QPushButton('Reset Camera')
        self.live_camera_reset.setToolTip('Discard this camera edit and restore its automatic or estimated pose.')
        self.live_camera_reset.clicked.connect(
            lambda: self.live_camera_edited(self.motion_view.selected_camera, None))
        footer.insertWidget(footer.count() - 1, self.live_camera_picker)
        footer.insertWidget(footer.count() - 1, self.live_camera_reset)
        self.viewport_checks['show_cameras'].toggled.connect(self.live_refresh_camera_controls)
        self.live_sync_camera_view()

    def live_camera_selected(self, index):
        if hasattr(self, 'live_camera_picker'):
            with QSignalBlocker(self.live_camera_picker):
                self.live_camera_picker.setCurrentIndex(index + 1)
            self.live_refresh_camera_controls()

    def live_refresh_camera_controls(self, *_):
        view = self.motion_view
        view.set_camera_editing(not (self.pear_input_pending or self.pear_align_phase))
        if not hasattr(self, 'live_camera_picker'):
            return
        visible = bool(view.input_cameras and view.show_cameras)
        self.live_camera_picker.setVisible(visible)
        self.live_camera_reset.setVisible(visible and view.selected_camera >= 0)
        self.live_camera_reset.setEnabled(view.camera_editing
                                          and view.selected_camera in self._live_camera_edits)
        if self._live_camera_edits:
            tip = 'Manual camera poses are active. Use Camera Align to solve on demand.'
        elif self.pear_auto_align.isEnabled():
            tip = 'Align automatically when starting tracking or when the input changes.'
        else:
            tip = 'Automatically align multiple camera inputs for fusion.'
        self.pear_auto_align.setToolTip(tip)
        view.update()

    def live_sync_camera_view(self):
        try:
            count = len(self.pear_sources())
            signature = self.pear_input_signature() if count else None
        except (ValueError, OSError):
            count, signature = 0, None
        signature_changed = signature != self._live_camera_signature
        if signature_changed:
            self._live_camera_signature = signature
            self._instant_camera_estimate = None
            self._live_camera_edits.clear()
            self.motion_view.select_camera(-1)

        recalibrated = self.pear_calibration != self._live_camera_baseline
        if recalibrated:
            self._live_camera_baseline = self.pear_calibration
            self._live_auto_cameras = []
            if self.pear_calibration is not None:
                self.live_read_aligned_cameras(count)

        cameras = []
        for index in range(count):
            if index < len(self._live_auto_cameras):
                pose, shape = self._live_auto_cameras[index]
                state = 'Aligned'
            else:
                if self._live_backend == 'instant' and count == 1:
                    center = self._instant_camera_estimate
                    pose = CameraPose(center if center is not None else np.array([0., 1.45, 3.2]), np.eye(3))
                else:
                    pose = assumed_camera(index)
                shape = (.24, .155)
                state = 'Estimated'
            if index in self._live_camera_edits:
                pose = self._live_camera_edits[index]
                state = 'Manual'
            cameras.append((pose, state, shape))
        self.motion_view.set_input_cameras(cameras)
        if hasattr(self, 'live_camera_picker'):
            with QSignalBlocker(self.live_camera_picker):
                self.live_camera_picker.clear()
                self.live_camera_picker.addItem('Select Camera…')
                for index, (_, state, _) in enumerate(cameras):
                    self.live_camera_picker.addItem(f'Camera {index + 1} · {state}')
                self.live_camera_picker.setCurrentIndex(self.motion_view.selected_camera + 1)
        self.live_refresh_camera_controls()
        if (signature_changed or (recalibrated and self._live_auto_cameras)) and cameras and self.motion_view.show_cameras:
            self.motion_view.frame_all()

    def live_read_aligned_cameras(self, count):
        try:
            metadata = json.loads((Path(self.pear_calibration[1]) / 'cameras.json').read_text('utf8'))
            extrinsics = np.asarray(metadata['extrinsics'], dtype=float)
            intrinsics = np.asarray(metadata['intrinsics'], dtype=float)
            sizes = np.asarray(metadata['sizes'], dtype=float)
            if (extrinsics.shape != (count, 3, 4) or intrinsics.shape != (count, 3, 3)
                    or sizes.shape != (count, 2) or not np.isfinite(extrinsics).all()
                    or not np.isfinite(intrinsics).all()
                    or (intrinsics[:, [0, 1], [0, 1]] <= 0).any()
                    or not np.isfinite(sizes).all() or (sizes <= 0).any()):
                raise ValueError('Camera data does not match the current inputs.')
            self._live_auto_cameras = [
                (CameraPose.from_extrinsic(matrix), (.21 * w / k[0, 0], .21 * h / k[1, 1]))
                for matrix, k, (w, h) in zip(extrinsics, intrinsics, sizes)]
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.log.append(f'Cannot display aligned cameras. {error}')

    def live_camera_edited(self, index, pose):
        if (self.pear_input_pending or self.pear_align_phase
                or not 0 <= index < len(self.motion_view.input_cameras)):
            return
        if pose is None:
            self._live_camera_edits.pop(index, None)
        else:
            self._live_camera_edits[index] = pose
        self.pear_generation += 1
        self._pear_last = None
        self._pear_world_anchor = None
        current = self.motion_view.input_cameras[index]
        if pose is None:
            self.live_sync_camera_view()
        else:
            self.motion_view.input_cameras[index] = (pose, 'Manual', current[2])
            if current[1] != 'Manual':
                self.live_camera_picker.setItemText(index + 1, f'Camera {index + 1} · Manual')
                self.live_refresh_camera_controls()
            else:
                self.motion_view.update()

    def load_pear_videos(self, paths):
        super().load_pear_videos(paths)
        self.live_refresh_camera_controls()

    def pear_inputs_failed(self, kind, token, error):
        super().pear_inputs_failed(kind, token, error)
        self.live_refresh_camera_controls()

    def pear_multi_invalidate(self):
        super().pear_multi_invalidate()
        self.live_sync_camera_view()

    def pear_backend(self):
        if self._live_backend == 'instant':
            from .instant_models import KEYS
            from .instant_runtime import status
            return 'InstantHMR', KEYS, 'instant_assets', 'instant_live', status
        return super().pear_backend()

    def pear_select_model(self):
        selected = self.model_choice.currentData()
        if selected != self._live_backend and selected in ('pear', 'instant'):
            if self.busy or self.contact_saving:
                with QSignalBlocker(self.model_choice):
                    self.model_choice.setCurrentIndex(self.model_choice.findData(self._live_backend))
                return True
            if self._pear_running or self.pear_align_phase:
                self.pear_toggle_tracking()
            self.accept_live_pose({}, None)
            self.pear_calibration = None
            self._live_backend = selected
            self._instant_camera_estimate = None
            self.live_sync_camera_view()
        return super().pear_select_model()

    def pear_should_auto_align(self):
        return super().pear_should_auto_align() and not self._live_camera_edits

    def pear_multi_request(self, value):
        super().pear_multi_request(value)
        if self._live_camera_edits and not value['align_job']:
            value['camera_edits'] = [
                self._live_camera_edits[i].extrinsic().tolist() if i in self._live_camera_edits else None
                for i in range(len(value['sources']))]

    def pear_begin_alignment(self):
        super().pear_begin_alignment()
        if self._live_backend == 'instant' and self.pear_align_phase == 'samples':
            self._pear_world_anchor = None
        self.live_refresh_camera_controls()

    def pear_cancel_alignment(self, message):
        super().pear_cancel_alignment(message)
        if hasattr(self, 'live_camera_picker'):
            self.live_sync_camera_view()

    def pear_alignment_failed(self, message):
        super().pear_alignment_failed(message)
        self.live_sync_camera_view()
        if self._live_camera_edits:
            self.pear_alignment_warning.setText(message + ' Tracking continues with manually placed cameras.')
        elif self._live_backend == 'instant':
            self.pear_alignment_warning.setText(message + ' Tracking continues in the selected camera view.')

    def pear_reset_bones(self):
        super().pear_reset_bones()
        self.live_sync_camera_view()

    def shutdown_preview(self):
        if getattr(self, '_live_shutdown', False):
            return
        self._live_shutdown = True
        super().shutdown_preview()

    def pear_sync_controls(self):
        super().pear_sync_controls()
        self.people_panel.show()
        instant = self._live_backend == 'instant'
        for widget in (self.pear_foot_lock, self.pear_reset_button):
            widget.setVisible(not instant)
        if instant:
            self.bone_overlay.setToolTip('Show the fixed MHR body and finger skeleton.')
            self.timeline_help_dialog.setText(
                'Open synchronized videos or assign webcams, then click Start Live Tracking. '
                'Use the frame ruler to inspect a video frame. Stop releases the tracking worker. '
                'The preview is not recorded or exported.')
            self.status.setText('InstantHMR · Body and finger tracking')
        for widget in (self.face_overlay, self.track_face, self.set_neutral_face,
                       self.reset_neutral_face, self.neutral_face_note,
                       self.rife_interpolation, self.reset_footlocks_button):
            widget.hide()
        self.follow_actor.setToolTip('Follow the captured person in the 3D view.')
        cameras = self.viewport_checks['show_cameras']
        cameras.setText('Cameras')
        cameras.setToolTip('Show and edit the input cameras in the 3D view.')
        cameras.show()
        self.motion_view.show_cameras = cameras.isChecked()
        self.live_refresh_camera_controls()

    def show_model_guide(self):
        if self._live_backend == 'instant':
            QMessageBox.information(self, 'InstantHMR Live',
                'Track one person from one or more fixed cameras using the MHR body and finger skeleton. '
                'Run Setup_InstantHMR.cmd with the app closed and prepare the files in Model Manager. '
                'Use Camera Align to fuse multiple views, or edit camera placement in the 3D view. '
                'Stream the skeleton to Blender, Unreal Live Link or a VMC receiver.')
            return
        QMessageBox.information(self, 'PEAR Live',
            'PEAR tracks one person from a webcam or video. Open Model Manager to prepare the tracking files. '
            'Use Camera Align for a shared floor and camera fusion. The Stream panel sends the live skeleton to Blender, Unreal or a VMC receiver.')

    def build_stream_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.stream_target = QComboBox()
        self.stream_target.addItems(['Blender', 'Unreal Live Link', 'VMC (Warudo)'])
        self.stream_target.setAccessibleName('Stream target')
        self.stream_target.currentIndexChanged.connect(self.refresh_live_stream)
        layout.addWidget(QLabel('Target'))
        layout.addWidget(self.stream_target)
        self.stream_port_controls = QWidget()
        row = QHBoxLayout(self.stream_port_controls)
        row.setContentsMargins(0, 0, 0, 0)
        self.stream_port = QSpinBox()
        self.stream_port.setRange(1024, 65535)
        self.stream_port.setValue(PORT)
        self.stream_port.setAccessibleName('Blender stream port')
        reset = QPushButton('Reset')
        reset.clicked.connect(lambda: self.stream_port.setValue(PORT))
        row.addWidget(self.stream_port)
        row.addWidget(reset)
        layout.addWidget(self.stream_port_controls)
        self.vmc_controls = QWidget()
        destination = QFormLayout(self.vmc_controls)
        destination.setContentsMargins(0, 0, 0, 0)
        self.vmc_host = QLineEdit(VMC_HOST)
        self.vmc_host.setAccessibleName('VMC destination IPv4 address')
        self.vmc_port = QSpinBox()
        self.vmc_port.setRange(1, 65535)
        self.vmc_port.setValue(VMC_PORT)
        self.vmc_port.setAccessibleName('VMC destination port')
        destination.addRow('IP address', self.vmc_host)
        destination.addRow('UDP port', self.vmc_port)
        layout.addWidget(self.vmc_controls)
        self.live_stream_help = QLabel()
        self.live_stream_help.setWordWrap(True)
        layout.addWidget(self.live_stream_help)
        self.live_stream_button = QPushButton('Start Stream')
        self.live_stream_button.clicked.connect(self.toggle_live_stream)
        layout.addWidget(self.live_stream_button)
        self.stream_status = QLabel()
        self.stream_status.setWordWrap(True)
        layout.addWidget(self.stream_status)
        layout.addStretch()
        self._stream_sender = None
        self._stream_error = None
        self._live_pose = LivePose()
        self._latest_pose = None
        self._published = False
        self._publish_times = deque()
        self._stream_timer = QTimer(page)
        self._stream_timer.setInterval(150)
        self._stream_timer.timeout.connect(self.refresh_live_stream)
        self._stream_timer.start()
        self.refresh_live_stream()
        return page

    def stream_tab_changed(self, index):
        # The live transport keeps running when focus moves to camera controls.
        return

    def suspend_stream(self):
        self.close_stream()

    def close_stream(self):
        sender, self._stream_sender = self._stream_sender, None
        if sender is not None:
            sender.close()
        self._published = False
        if hasattr(self, '_publish_times'):
            self._publish_times.clear()
        if hasattr(self, 'live_stream_button'):
            self.live_stream_button.setText('Start Stream')
            self.stream_target.setEnabled(True)
            self.stream_port_controls.setEnabled(True)
            self.vmc_controls.setEnabled(True)

    def toggle_live_stream(self):
        self._stream_error = None
        if self._stream_sender is not None:
            self.close_stream()
        else:
            try:
                target = self.stream_target.currentIndex()
                if target == 0:
                    self._stream_sender = BoneStream(self.stream_port.value())
                elif target == 1:
                    self._stream_sender = UnrealStream(ROOT)
                else:
                    self._stream_sender = VmcStream(self.vmc_host.text(), self.vmc_port.value())
            except (OSError, RuntimeError, ValueError) as error:
                self._stream_error = str(error)
            else:
                self.live_stream_button.setText('Stop Stream')
                self.stream_target.setEnabled(False)
                self.stream_port_controls.setEnabled(False)
                self.vmc_controls.setEnabled(False)
                now = monotonic()
                if self.pose_is_available(now):
                    self.publish_live_pose(self._latest_pose[2], now)
        self.refresh_live_stream()

    def pose_is_paused_video(self, metadata):
        return metadata['mode'] == 'video' and not self.timer.isActive()

    def pose_is_available(self, now):
        if (self._latest_pose is None or not self._pear_running or self.pear_align_phase
                or self._pear_failed or self.pear_input_pending or self.pear_switching):
            return False
        generation, stamp, (metadata, _) = self._latest_pose
        if generation != self.pear_generation:
            return False
        if self.pose_is_paused_video(metadata):
            return metadata['timeline_frame'] == self.video_view.timeline_requested_index
        return now - stamp <= STALE_SECONDS

    def release_live_pose(self):
        if self._stream_sender is not None and self._published:
            self._stream_sender.publish((), ())
        self._published = False
        self._publish_times.clear()

    def accept_live_pose(self, metadata, pose):
        if self._live_backend == 'instant' and 'camera_center' in metadata:
            center = np.asarray(metadata['camera_center'], dtype=float)
            if center.shape == (3,) and np.isfinite(center).all() and not np.array_equal(center, self._instant_camera_estimate):
                self._instant_camera_estimate = center
                self.live_sync_camera_view()
        if pose is None:
            self._latest_pose = None
            self.release_live_pose()
            return
        # Avoid converting packets until a sender is active. Retain the latest
        # completed packet so a sender started mid-capture can publish at once.
        now = monotonic()
        stamp = metadata.get('timestamp', now) if metadata['mode'] == 'camera' else now
        self._latest_pose = (metadata['generation'], stamp, (metadata, pose))
        if self._stream_sender is not None and self.pose_is_available(now):
            self.publish_live_pose(self._latest_pose[2], now)

    def refresh_live_stream(self):
        target = self.stream_target.currentIndex()
        self.stream_port_controls.setVisible(target == 0)
        self.vmc_controls.setVisible(target == 2)
        help_text = (
            'Connect the RGB MoCap Blender add-on to 127.0.0.1.',
            'Run Setup_UnrealStream.cmd, then add RGB MoCap as a Message Bus source in Unreal.',
            'In Warudo, set up VMC pose tracking on the character and match its receiver port.',
        )
        self.live_stream_help.setText(help_text[target])
        sender = self._stream_sender
        if sender is None:
            self.stream_status.setText(self._stream_error or 'Stream stopped.')
            return
        now = monotonic()
        available = self.pose_is_available(now)
        if not available:
            self.release_live_pose()
        else:
            while self._publish_times and self._publish_times[0] < now - 1:
                self._publish_times.popleft()
        if isinstance(sender, VmcStream):
            sender.refresh()
            connection = f'VMC UDP → {sender.destination[0]}:{sender.destination[1]}'
        elif isinstance(sender, UnrealStream) and not sender.ready:
            connection = 'Starting Unreal Live Link sender…'
        else:
            connection = 'Receiver connected' if sender.connected else 'Waiting for receiver'
        error = sender.error()
        if error and not isinstance(sender, VmcStream):
            self._stream_error = error
        if error or self._stream_error:
            self.stream_status.setText(error or self._stream_error)
            return
        if not available:
            self.stream_status.setText(f'{connection}\n{LABEL} · Waiting for {self.pear_backend()[0]} pose.')
        elif self.pose_is_paused_video(self._latest_pose[2][0]):
            self.stream_status.setText(f'{connection}\n{LABEL} · Holding current frame')
        else:
            age = round((now - self._latest_pose[1]) * 1000)
            self.stream_status.setText(f'{connection}\n{LABEL} · {len(self._publish_times)} poses/s\nPose age {age} ms')

    def publish_live_pose(self, sample, now):
        metadata, pose = sample
        try:
            packet = self._live_pose.packet(self.pear_session, metadata, pose)
            if isinstance(self._stream_sender, VmcStream):
                self._stream_sender.publish((packet,), (ACTOR,), normalize_arms=self._live_backend == 'instant')
            else:
                self._stream_sender.publish((packet,), (ACTOR,))
        except (KeyError, TypeError, ValueError) as error:
            self._stream_error = str(error)
            self.release_live_pose()
            return
        self._published = True
        self._publish_times.append(now)
        while self._publish_times and self._publish_times[0] < now - 1:
            self._publish_times.popleft()
