# RGB MoCap Connect

Connect the RGB MoCap Live **Person 1** source to a MetaHuman or UE5
Manny/Quinn actor. The plugin uses the meshes from your Unreal project.

## Install

Copy `RGBMoCapMetaHuman` under your Unreal project's `Plugins` and build the
project's Editor target. Enable **RGB MoCap Connect**, **Live Link** and
**UDP Messaging**. Follow the [sender guide](../../docs/UNREAL_LIVE_LINK.md).

## Connect

Open **Tools > RGB MoCap Connect**. Set the target type, select the source
and actor, and click **Connect selected actor**. The target uses Animation
Blueprint mode and positive uniform scale. **Ground Contact** and **Floor
Offset** operate on an estimated flat floor in component space. Live's
source carries body and finger motion.

## Build and verification

Build with the same engine and toolchain as the receiving project.
See [receiver validation](VALIDATION.md) and [security](SECURITY.md).
The authored source uses the MIT License.
