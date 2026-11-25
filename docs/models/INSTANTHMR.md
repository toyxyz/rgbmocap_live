# InstantHMR in RGB MoCap Live

InstantHMR is an optional single-person tracker for fixed webcams or
synchronized videos. It produces a live 52-bone body and finger pose. The model runs on Windows
x64 with a compatible NVIDIA CUDA GPU.

## Set up

1. Run the Live app's `install.cmd` to prepare the base runtime.
2. Close RGB MoCap Live and run `Setup_InstantHMR.cmd`. The setup downloads
   and verifies pinned ONNX Runtime GPU packages and runs a CUDA execution test.
3. Launch `RGBMoCapLive.cmd`, choose **InstantHMR · Live** and open **Model
   Manager**. In the **InstantHMR** tab, use **Auto Download** or **Import** for
   **InstantHMR Medium Commercial**, **MHR Body Graph**, and **Person Detection /
   YOLOv8s** if missing. The detector is shared with PEAR.
4. Open one or more videos, or assign one webcam per camera slot. Then click
   **Start Live Tracking**. Use **Stop** to unload the tracker. Use **PEAR · Live**
   when the workflow needs foot lock.

The stream sends **Person 1** to [Blender](../BLENDER_STREAM.md),
[Unreal Live Link](../UNREAL_LIVE_LINK.md), or
[VMC receivers such as Warudo](../VMC_STREAM.md). Scrubbing estimates the selected
frame and video playback follows the latest completed frame. Switching models
stops tracking and clears the old stream pose. The stream can remain connected.

## Pose and placement

InstantHMR estimates pelvis translation and joint rotations from the camera
view. **World Position** controls whether the pelvis can move. The translation
uses an assumed focal length and an initial floor estimate before alignment.
**One Euro Filter** can smooth rotations and translation independently.
InstantHMR streams body and fingers. The live workspace has no face capture,
recording or animation export.

## Camera alignment

Prepare the optional camera runtime with the app closed by running
`install.cmd --with-camera-align`. Download **VGGT-Omega** and **GeoCalib**
from **Model Manager > InstantHMR**. These files are shared with PEAR.
Use fixed cameras with overlapping coverage of the same performer and floor.
For video, open clips together with synchronized starts and choose a frame with
at least one later frame in the active trim. Click **Camera Align** to solve on
demand. **Auto Camera Align** defaults off for one input and turns on for
multiple inputs. **VGGT-Ω OPT** reduces peak GPU memory.

Use **Cameras** in the 3D viewport to show the camera. Select it and drag an
axis arrow or a rotation ring. **Reset Camera** returns to the solved or
estimated pose. Manual edits work with or without the automatic alignment
models. Estimated cameras join the fusion after alignment or manual placement.
Tracking falls back to the selected view if automatic alignment cannot solve.
Webcams use arrival-time synchronization. Keep one performer visible across
the views so the per-camera trackers follow the same person.

The shared fusion solver combines joint rotations using visibility weights
and estimates one pelvis position from the calibrated views. The result is
one fixed-length body and finger pose in the viewport and stream. A view with
no usable person observation contributes no weight to the pose.

## Model

The pinned Medium Commercial network runs on ONNX Runtime CUDA. Live extracts
only the pre-skinning FK state from the pinned MHR Body Graph, skips body mesh
evaluation and maps the native joints to fixed-length body and finger bones.
The VMC sender converts MHR's arm reference to a T-pose for normalized avatars.

The upstream weights are downloaded separately. Review the model terms before
obtaining or using them. Upstream identifies both this network and the shared
MHR ONNX graph as subject to the SAM License. The Medium Commercial checkpoint
is the variant without the additional non-commercial training-image restriction
listed for other sizes. [Official InstantHMR license notes](https://github.com/Naturalpad-studio/InstantHMR/blob/6b187eada5cf1deac6a2eaa1a85edcad7bfba660/README.md#licence) ·
[Official model files](https://huggingface.co/momolesang/InstantHMR/tree/fcde80ffe1181c946ef99251484c80083e8faede)
