# Stream to Warudo or another VMC receiver

## Connect

1. In Warudo, load a humanoid avatar with a normalized T-pose. Use
   **Character > Setup Motion Capture** and choose **VMC** for pose tracking.
   The setup connects the receiver to the character through its pose graph.
2. In RGB MoCap Live, choose **Stream > Target > VMC (Warudo)**. For the
   same PC, use **IP address** `127.0.0.1` and **UDP port** `39539`.
   If the receiver runs on another PC, enter its IPv4 address and matching
   port. The VMC port is independent of the Blender stream port.
3. Click **Start Stream** and **Start Live Tracking**. A paused video holds
   its current pose. **Stop Stream** or loss of the current track reports
   tracking as unavailable. The receiver decides how to display an
   unavailable pose.

VMC sends UDP datagrams without a receiver handshake. The app displays the
destination and the outgoing pose rate while the track is playing.

## Body and rest pose

The stream covers one person and 52 Unity `HumanBodyBones`. It includes the
body, toes and thirty finger bones. The active model's preview pose supplies
the output, including the One Euro filter when enabled. PEAR also applies its
enabled camera alignment and foot lock. Face expression, jaw and eye channels
are outside the body feed.

Bone transforms use meters and Unity's Y-up coordinates. The source bone's
calibrated rest axes are removed before computing parent-relative rotations.
The normalized neutral pose sends identity quaternions. InstantHMR converts
the MHR arm reference to a T-pose for normalized receivers. Rotations retain
the tracked thumb-base axial movement. Non-pelvis translation fields contain
fixed rest offsets.
Pelvis translation is sent in `Hips` and the VMC model root stays at identity.
For receivers with a bone position synchronization option, disable that
option if you want the avatar to retain its own limb lengths. This mode
targets normalized humanoids. An avatar with baked bone rotations may require
normalization or avatar-specific retargeting in the receiving software.

The sender uses OSC bundles over UDP and publishes `/VMC/Ext/Root/Pos`,
`/VMC/Ext/Bone/Pos`, `/VMC/Ext/T` and `/VMC/Ext/OK`.

[VMC protocol specification](https://protocol.vmc.info/english) ·
[Warudo VMC setup](https://docs.warudo.app/docs/mocap/vmc) ·
[Warudo pose setup overview](https://docs.warudo.app/docs/mocap/overview) ·
[Unity HumanBodyBones](https://docs.unity.com/en-us/engine/6000.0/script-reference/unityengine/humanbodybones) ·
[OSC 1.0 specification](https://opensoundcontrol.stanford.edu/spec-1_0.html)
