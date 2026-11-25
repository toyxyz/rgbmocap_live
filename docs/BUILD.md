# Develop RGB MoCap Live

Run `install.cmd` in the Live checkout and start `RGBMoCapLive.cmd`. Add
`--with-camera-align` during setup for the camera solver. Use
`Setup_InstantHMR.cmd` for the optional ONNX CUDA overlay.

| Area | Source |
| --- | --- |
| Desktop and stream lifecycle | `mocap/live_window.py` and `mocap/live_pose.py` |
| Inference worker and bounded mailbox | `mocap/pear/worker.py` and `mocap/pear/session.py` |
| Capture and alignment | `mocap/pear/camera_preview.py` and `mocap/pear/align_worker.py` |
| Multi-camera fusion | `mocap/pear/multi_worker.py` and `mocap/pear/multiview.py` |
| InstantHMR | `mocap/instant_live.py` and `mocap/instant_alignment.py` |
| Base installer | `tools/git_managed_install.py` and `installer/runtime-lock.json` |

The [Unreal components](UNREAL_LIVE_LINK.md) use the target Engine toolchain.
The [Blender receiver](BLENDER_STREAM.md) runs in Blender.

Test the normal launcher, the affected worker and the receiving transport.
Distinguish video, simulated webcams and physical camera tests. Consult the
[validation record](LIVE_VALIDATION.md) and [agent instructions](../AGENTS.md).
Keep models, runtimes, calibration samples, settings and test media local.
