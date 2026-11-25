# Stream RGB MoCap Live to Unreal Engine

[RGB MoCap Connect](../unreal_plugin/RGBMoCapMetaHuman/README.md) maps the
**Person 1** source to an assembled MetaHuman or a UE5 Manny/Quinn actor.

## Build the sender locally

Complete `install.cmd` and close Live. Install Unreal Engine 5.6 or later,
Visual Studio C++ tools and a Windows SDK. From the Live checkout run the
following in PowerShell, adjusting the engine path.

    .\Setup_UnrealStream.cmd --engine "C:\Program Files\Epic Games\UE_5.8"

Setup compiles and verifies the sender. Repeat after changing the Engine
build or sender source.

## Prebuilt downloads

If you have a sender ZIP built for this source and the exact Engine BuildId,
run `.\Setup_UnrealStream.cmd --prebuilt "C:\Downloads\RGBMoCapSender.zip"`.
Add `--engine` for a custom Engine path. This source checkout includes the
sender and receiver code.

## Connect

1. Build the receiving plugin if using RGB MoCap Connect and enable
   **Live Link** and **UDP Messaging** in the Unreal project.
2. In Live, start tracking, select **Unreal Live Link** in **Stream**, and
   click **Start Stream**.
3. In Unreal, use **Window > Virtual Production > Live Link** and choose
   **Add Source > Message Bus Source > RGB MoCap**. The subject starts with
   `RGBMoCap_Person_1_`.
4. In **Tools > RGB MoCap Connect**, choose a target type, select the source
   and actor, then click **Connect selected actor**. Use a Realtime viewport
   or Play/Simulate.

Live sends body and finger animation. If reception stalls, restart the
stream and check the Live Link status and Engine BuildId.
