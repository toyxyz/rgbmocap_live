# RGB MoCap Live agent guide

This repository is the independent Windows live app for PEAR and InstantHMR. Start with the [Live user guide](docs/USER_GUIDE.md) and [development guide](docs/BUILD.md).
The original RGB MoCap project is a separate repository and is a read-only
reference for work on this app.

- Entry points are `RGBMoCapLive.cmd` and `launch_live.py`. Build and repair
  the private runtimes with `install.cmd`. Camera Align is optional and uses
  `install.cmd --with-camera-align`.
- Keep capture, model inference and camera alignment in separate
  processes. Keep pending frames and outbound snapshots bounded. Route a
  pose to the stream only after its generation is accepted by the viewport.
- Live fixed-rig PEAR skips the position map, UV rasterization and head UV
  lookup tables through the pinned adapter. Preserve SMPL-X/FLAME joint and
  surface data for initial calibration and Reset Bones. Adapter changes require
  regenerating the support profile through the installer.
- Auto Camera Align defaults off for one input and turns on for multiple
  inputs. Keep the single-input preference local. The inference worker pauses with
  its model resident while the separate alignment worker runs and resumes on
  a new request. Failed alignment must allow selected-camera or manual-camera
  tracking.
- [Manual placement](docs/models/PEAR.md#manual-camera-placement) is Live-only.
  Keep camera edits scoped to the active source signature and pose generation.
  Apply edited extrinsics in the worker. Preserve solved intrinsics and the
  original calibration file, and exclude estimated cameras from manual fusion
  until they have been placed.
- Camera Align must retain VGGT depth and confidence in both modes. Keep
  the default-on OPT setting local to this app and fall back to standard VGGT
  when native CUDA BF16 is unavailable.
- `mocap/live_pose.py` adapts the calibrated body and finger hierarchy to
  the existing Blender and Unreal receiver contracts. Publish the single
  subject as Person 1 with a stable actor ID. Hold an accepted paused video
  pose only for its current frame and generation. Preserve coherent rest and
  active rotations, units, root translation and loss-of-track handling.
- `mocap/vmc_stream.py` sends the [normalized VMC humanoid](docs/VMC_STREAM.md).
  Remove the calibrated rest basis before deriving parent-relative rotations.
  Keep Unity axes and bone names consistent, retain thumb-base roll, and send
  actor translation once on Hips. UDP status must distinguish sending from
  receiver verification.
- Local models, runtimes, caches, input videos and settings are user assets.
  Keep them out of Git and release artifacts. Verify staged paths and promote
  repaired runtimes through the existing install transaction.
- [InstantHMR](docs/models/INSTANTHMR.md) runs in the shared live inference worker
  under its own stage and optional ONNX Runtime CUDA overlay. Keep the network
  and MHR graph pinned. Extract native joint state before the skinning stage.
  Preserve the fixed 52-channel rig, source-relative rotations and MHR-to-VMC
  arm rest conversion. InstantHMR supplies MHR observations to the same
  camera solver and multiview fusion path as PEAR. Preserve the four-view
  inference batch limit, fixed MHR offsets, rotation averaging and weighted
  pelvis solve. Preserve the resident worker during alignment and scope solved
  camera data to the active model. Foot lock belongs to PEAR. Reuse the shared
  YOLOv8s model catalog entry.
- Before claiming compatibility, test the selected real model, stream transport,
  and the named receiver configuration. Report test coverage and limitations.
