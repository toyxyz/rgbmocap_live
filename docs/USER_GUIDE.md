# RGB MoCap Live user guide

[Overview](../README.md) · [Development](BUILD.md) · [Validation](LIVE_VALIDATION.md)

## Install and update

Use Windows x64, a compatible NVIDIA GPU and driver, and a writable folder.
Initial setup needs internet access and disk space for a private Python
runtime. Clone this repository or extract its source ZIP.

    git clone https://github.com/toyxyz/rgbmocap_live.git

Run `install.cmd` from the Live folder. Use `install.cmd --with-camera-align`
for the optional automatic camera solver. For InstantHMR, run
`Setup_InstantHMR.cmd` after the base install. Start `RGBMoCapLive.cmd`.

Close Live before setup or updates. Pull new source in a Git checkout, or
extract a new source ZIP to a separate folder. Run `install.cmd` when the
launcher requests a runtime repair after a source update.

## Prepare the models

- **PEAR · Live** uses the PEAR checkpoint, original geometry bundle and
  YOLOv8s. See [PEAR setup](models/PEAR.md#setup).
- **InstantHMR · Live** uses the Medium Commercial network, MHR body graph and
  the same detector. See [InstantHMR setup](models/INSTANTHMR.md#set-up).
- **Camera Align** uses VGGT-Omega and GeoCalib. Both workflows share these
  files. Manual camera placement works with the tracking files alone.

## Track and align

Open a video or choose a webcam, then click **Start Live Tracking**. For
several views, use fixed cameras with overlapping coverage of the same
performer and floor. Open clips with synchronized starts together, or assign
a distinct webcam to each slot. Video thumbnails change the preview camera.
Use the timeline to play or seek. **World Position** controls pelvis travel.
**One Euro Filter** smooths rotations and translation. **Stop** unloads the
tracking worker.

**Auto Camera Align** defaults off for one view and turns on for multiple
views. Click **Camera Align** to solve on demand. For video, leave at least one
later frame in the trim. Show **Cameras** in the 3D viewport, select a camera,
and drag arrows to move or rings to rotate it. **Reset Camera** restores its
estimated or solved pose. If automatic alignment fails, tracking can continue
from the selected view or from manually placed cameras.

## Stream

Choose a destination in **Stream** and click **Start Stream**. Tracking and
streaming have separate controls. See [Blender](BLENDER_STREAM.md),
[Unreal Live Link](UNREAL_LIVE_LINK.md), or [VMC and Warudo](VMC_STREAM.md).

## Troubleshooting

- Prepare the active workflow's files in **Model Manager**.
- For Camera Align, prepare its optional runtime and both camera files.
- Close other camera apps if a webcam is unavailable. Webcam synchronization
  uses frame arrival times. Use synchronized clips for fixed timing.
- For path errors, use a short writable folder such as `C:\Apps\rgbmocap_live`.
- For runtime repair, close Live and run `install.cmd --verify-all`. Check the
  console and `.setup/logs` if setup fails again.
