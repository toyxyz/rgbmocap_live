# Stream to Blender

1. In Blender 4.5 or later, install
   [the RGB MoCap Bone Stream ZIP](../blender_addon/rgbmocap_bone_stream.zip)
   through **Edit > Preferences > Add-ons > Install from Disk**, then enable it.
2. In Live, select **Blender** in the **Stream** tab and click **Start Stream**.
   Use port `49168` on `127.0.0.1` unless you update the port on both sides.
3. In the Blender 3D Viewport sidebar, open **RGB MoCap** and connect.
   Start Live tracking to receive the **Person 1** body and finger rig.

A paused video holds the current pose. Stopping tracking clears the pose.
