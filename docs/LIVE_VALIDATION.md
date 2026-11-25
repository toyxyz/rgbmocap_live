# Initial Windows validation

- The independent installer verified the shared, GUI and optional camera
  runtime profiles and retained the PEAR and Camera Align support overlays.
- The live workspace opened with PEAR as the sole model. Model Manager exposed
  the three tracking files and the two optional Camera Align files.
- A 720 × 1280, 30 fps video produced 20 live stream packets
  with a 52-bone schema. A stop command published an empty snapshot and the
  application exited normally.
- The existing Blender receiver running inside Blender 4.5.0 created one
  52-bone armature, including the finger bones. Its Hips transform changed
  across eight received poses.
- The app-local Unreal Engine 5.8 commandlet completed its native provider
  handshake while receiving the active live pose from the app.
- An external Unreal receiver driving MetaHuman or Manny/Quinn, physical
  webcam input, and continuous long-duration latency were not measured
  in this initial video test.

## VGGT-Omega OPT for Camera Align

Windows, RTX 5090, VGGT-Omega 512 checkpoint. Each pair used the same images,
PEAR observations and rig. Camera Align captures two samples per view.
Values are peak PyTorch GPU allocations for the VGGT stage, including model load
and camera and depth inference. GeoCalib ran afterward.

| Input | Standard | OPT | Reduction |
| --- | ---: | ---: | ---: |
| One video view, two samples | 6.08 GiB | 2.88 GiB | 53% |
| Three synchronized video views, six samples | 6.37 GiB | 2.96 GiB | 54% |

The app completed automatic alignment and resumed PEAR tracking with OPT on
both inputs. A second run with OPT off completed the standard alignment for
each saved sample set. The estimated camera rotations differed by at most
0.08°, fitted camera centers by at most 8.2 mm, and fitted scale by less than
0.04% across the two tests. Differences are relative to standard VGGT. The
tests used prerecorded input and did not measure webcam performance.

## Auto Camera Align and resident PEAR

On Windows with an RTX 5090, one prerecorded view started tracking with
Auto Camera Align off and created no alignment worker. Manual alignment
worked with the toggle off. Cancelling an alignment preserved the PEAR
process and resumed the selected view. Reset Bones with the toggle off
updated the rig and left camera alignment available as a manual action.

Enabling automatic alignment on the single view triggered it without
restarting PEAR. Switching to three synchronized videos enabled automatic
alignment and produced fused poses from all three views using the same PEAR
process. Returning to one video restored the saved single-input preference.

A separate allocator-cap test triggered a CUDA out-of-memory error in the
alignment worker. The app reported the failure and resumed unaligned local
Blender transport from the same PEAR process. These checks used prerecorded
video inputs.

## Automatic Person 1 streaming

A paused video pose stayed available after the normal live freshness window.
Blender 4.5.0, running the packaged receiver, connected to the paused stream,
created the 52-bone Person 1 armature in slot 1, and received the automatic
selection. Seeking updated the same actor. The Unreal Engine 5.8 provider
completed its handshake, and the outgoing subject used the
`RGBMoCap_Person_1_` prefix. Stopping PEAR cleared the active stream pose.

## Fixed-rig PEAR startup

The live PEAR worker skips the position map, UV interpolation maps and head UV
indices in its pinned SMPL-X/FLAME adapter. A full-geometry constructor still
creates all of them by default.

On an RTX 5090, the previous single-video path took 17.7 to 19.2 seconds to
display the first pose with automatic camera alignment off. Two runs through
the installed update took 10.3 and 10.7 seconds. The measured geometry
constructor took 2.1 to 2.4 seconds, down from 9.4 to 10.6 seconds. Compared
arrays for three frames and Reset Bones were bitwise equal, including rest
bone geometry, joints and rotations. The regenerated support profile passed
CUDA verification, and three prerecorded views completed automatic alignment
and resumed fused tracking in the same PEAR worker.

## Manual camera placement

On Windows with GPU PEAR and the native OpenGL viewport, selecting the camera
label exposed move arrows and rotation rings together. In a paused single-view
video, dragging the translation handle moved the pelvis by the camera offset.
The pelvis orientation followed a separate rotation drag. Escape reverted a
pending edit, and Reset Camera restored the estimated pose. Editing kept the
same PEAR process and fixed-rig calibration.

With three synchronized videos, the displayed camera poses matched the
automatic calibration. Moving one camera kept the other two in fusion and
preserved the calibration file. Reset Camera restored the automatic extrinsics.
The loopback Blender-stream packet for Person 1 contained the edited root
position. Replacing the inputs cleared the camera overrides. These checks
used prerecorded video and a local protocol receiver.

## Tracking throughput after camera alignment

After warmup, one prerecorded 30 fps view delivered about 30 pose fps with
or without manual camera alignment. In a separate replay through the shared
webcam-frame transport, the calibrated worker delivered 29 poses/s and spent
about 19 ms waiting for a fresh input frame. The previous timing included that
wait only on the calibrated path. The status now gives processing time and
observed preview pose rate separately.

On three synchronized 30 fps views, replaying the same automatic calibration
raised the observed preview rate from 25.1 to 27.7 pose fps. Median frame
processing fell from 35.0 to 31.5 ms. Fusion fell from 4.6 to 1.7 ms and
detection plus PEAR inference remained about 20 ms. A deterministic replay
using the same rig and camera calibration verified unchanged float32 joints,
bounds, rotations, and bone-facing outputs for 160 frames each with one and
three cameras, including intermittent loss of views. These measurements use
prerecorded video and shared-memory replay.

## Live pipeline audit

On an RTX 5090, three synchronized 30 fps recordings processed in about
30 ms per frame even when only one view supplied a pose. Selecting only the
views that contribute to the pose reduced the median to 20.4 ms for the
selected-camera fallback and 20.9 ms for one manually placed camera. All
three views together took 31.1 ms. The principal cost in that mode was
person detection plus PEAR at about 19.7 ms. Decode took 5.7 ms, the local
pose and preview pipe took 5.2 ms, and the next UI poll took 9.3 ms
(medians at separate pipeline boundaries).

While the inference worker delivers current webcam images, the capture
worker publishes raw frames to shared memory and sends only lightweight
preview metadata or thumbnails to the GUI. The separate full-size preview
returns when needed. A 1280 by 720 capture simulation verified removal of
2,764,800 duplicate RGB bytes per capture-preview packet.

The live worker recovered after person loss and an expired Reset Bones
request. A bad camera observation was excluded from fusion without losing
valid views. Tests also covered a long-to-short clip switch with automatic
alignment, a duplicate reopen, a failed input inspection, mixed-rate final
frames, and edited Person 1 packets over the local stream. Tests used
prerecorded input and shared-memory replay.

## Installed Live app review

The Windows launcher opened and closed normally, and a second instance was
rejected by the installation lock. The three installed runtime groups passed
38,337 file integrity and import checks. CUDA matrix multiplication passed
on the RTX 5090, and all five Live model entries matched their pinned hashes.

The Live Model Manager showed the five available items and handled malformed
model and geometry receipts as unavailable. Malformed optional alignment
metadata returned a setup repair error. Official YOLOv8s import, managed
removal, and re-import passed in an isolated model directory.

Single-view preview, three-view automatic alignment, manual camera edits,
input changes, and the shared-memory webcam replay passed. A 45-second
three-view loop produced 1,166 preview poses and about 3 MB of local stream
traffic. Closing the app reaped the tracking worker. The packaged add-on
received Person 1 in Blender 4.5.0 and preserved its 52-bone rig while pausing
and seeking. The Unreal Engine 5.8.4 provider started, accepted the Person 1
snapshot, and cleared it when tracking stopped.

The Live-focused unit selection passed all 18 tests. The inherited full
suite reported four failures and 24 error entries in the offline defaults,
FlowHMR intrinsics, and multi-workflow Model Manager tests. The review used
the existing Windows installation, recorded video, and shared-memory webcam
replay. Unreal testing covered the app sender and provider, with actor
driving and physical webcam hardware outside the review.

## VMC stream

The normalized reference sent identity quaternions for all 52 Unity bones.
An independent UDP decoder verified parent-relative rotations, hips position,
finger mappings, thumb-base roll, pause keepalives, and status transitions.
A real PEAR video run sent 79 active VMC snapshots in three seconds. Seeking
changed the body and finger pose, and stopping tracking sent an inactive
status with no bone transforms. A 52-bone bundle took about 0.07 ms to encode
on the test PC.

The installed Warudo instance listened on UDP 39539. Avatar motion in that
scene was outside this verification. The packaged receiver in Blender 4.5.0
and the UE 5.8.4 Live Link sender passed their existing transport checks with
VMC installed as a separate target.

## InstantHMR live tracking

A 1080 × 1920, 30 fps video on Windows with an RTX 5090 was used to verify the
InstantHMR Medium Commercial pipeline. CUDA runtime setup executed an ONNX
operator on the GPU. Both models downloaded and imported successfully. An
incorrect graph was rejected without replacing the imported MHR model.

- The viewport presented a 52-bone pose. Seeking updated the body and fingers.
  One three-second playback measured a 25.54 ms median per processed frame
  and 70 displayed poses. The initial test started in about 50 seconds, and
  a later start took about 8 seconds.
- The Blender 4.5 receiver created the 52-bone rig and followed frame seeks.
  The Unreal Engine 5.8 sender published the Person 1 subject and cleared it
  when tracking stopped.
- A VMC loopback captured all 52 channels and reconstructed the mirrored
  preview joint positions from the transmitted local transforms. Switching to
  PEAR cleared the previous pose and resumed after explicitly starting PEAR.
- Physical webcam input and avatar motion inside Warudo and Unreal Engine
  were outside this video validation.

### InstantHMR single-camera alignment

- An actual 1080 × 1920 video completed the two-sample VGGT-Ω OPT and GeoCalib
  solve in 24.7 seconds on the RTX 5090. The same InstantHMR worker resumed
  after alignment and the viewport camera matched the solver's extrinsic matrix.
- Translation and rotation from manual camera edits changed the paused pose.
  The VMC loopback carried the edited pelvis position. Reset restored the
  estimated camera before alignment and the solved camera after alignment.
- A webcam capture replay through Windows shared memory completed alignment
  with the same tracking and capture processes. A simulated missing-alignment
  error resumed the unaligned pose. The test used video replay for camera
  frames.

### InstantHMR multi-camera fusion

- Three synchronized real videos completed automatic VGGT-Ω OPT and GeoCalib
  alignment. The InstantHMR worker stayed resident, used all three views, and
  sent one **Person 1** snapshot over the Blender transport. The received
  pelvis matched the viewport pose.
- The viewport supported editing and resetting aligned cameras. Changing
  inputs cleared the prior camera edits, and returning to single-view tracking
  kept the same inference process.
- A numerical check with the official MHR bind skeleton exercised three, two,
  one, and zero valid camera observations. The recovered pelvis, joint positions
  and bone rotations matched the synthetic reference, and zero observations
  yielded no pose before tracking recovered. Physical multiview webcam input
  was outside this video validation.
