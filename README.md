# RGB MoCap Live

Track one performer's body and fingers from webcams or synchronized videos on
Windows. PEAR and InstantHMR can combine multiple views into one 52-bone pose.
Preview the skeleton and stream **Person 1** to Blender, Unreal Live Link, or
a VMC receiver such as Warudo.

1. Clone the repository or extract its source ZIP into a short writable path
   on Windows x64 with a compatible NVIDIA GPU and driver.
2. Run `install.cmd`, then `RGBMoCapLive.cmd`.
3. Prepare the selected model's files in **Model Manager**. For InstantHMR,
   first run `Setup_InstantHMR.cmd` with Live closed.
4. Open a clip or assign a webcam and click **Start Live Tracking**. For
   multiple views, open synchronized clips together or assign distinct webcams.
   **Camera Align** estimates the camera poses and floor.
5. Choose a destination in **Stream** and click **Start Stream**.

| Guide | What it covers |
| --- | --- |
| [Live user guide](docs/USER_GUIDE.md) | Installation, tracking, cameras and troubleshooting |
| [PEAR](docs/models/PEAR.md) | Model setup, multi-camera tracking and Foot Lock |
| [InstantHMR](docs/models/INSTANTHMR.md) | CUDA runtime, MHR models and multi-camera tracking |
| [Blender](docs/BLENDER_STREAM.md) | Local receiver add-on |
| [Unreal Live Link](docs/UNREAL_LIVE_LINK.md) | Sender, MetaHuman and UE5 mannequin setup |
| [Warudo and VMC](docs/VMC_STREAM.md) | UDP receiver setup |
| [Development](docs/BUILD.md) and [validation](docs/LIVE_VALIDATION.md) | Code entry points and tested configurations |

Live keeps its installer, models, settings and cache in the Live folder.
Video playback and seeking drive the stream. Pausing holds the current
accepted pose. Stopping tracking clears the active pose.

## License

Authored code and documentation use the [MIT License](LICENSE). Model files
are obtained separately under their providers' terms. See
[third-party notices](THIRD_PARTY_NOTICES.md) and [citations](docs/CITATIONS.md).
