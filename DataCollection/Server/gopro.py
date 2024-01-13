import asyncio
from open_gopro import WirelessGoPro, Params


async def main(commandqueue):
    async with WirelessGoPro() as gopro:
        
        
        await gopro.ble_setting.resolution.set(Params.Resolution.RES_4K)
        await gopro.ble_setting.fps.set(Params.FPS.FPS_30)
        
        command = commandqueue.get()

        if command == 'start':
            print("GoPro recording started!")
            await gopro.ble_command.set_shutter(shutter=Params.Toggle.ENABLE)

            # Wait for the next command
            next_command = commandqueue.get()
            
            if next_command == 'stop':
                print("GoPro recording stopped!")
                await gopro.ble_command.set_shutter(shutter=Params.Toggle.DISABLE)
                print("Downloading file..")
                media_list = (await gopro.http_command.get_media_list()).data.files
                await gopro.http_command.download_file(camera_file=media_list[-1].filename)

            
        # end = input("Press enter to stop recording..")
        # # await asyncio.sleep(2) # Record for 2 seconds
        # await gopro.ble_command.set_shutter(shutter=Params.Toggle.DISABLE)
        # print("Recording stopped! Downloading file...")
        # # Download all of the files from the camera
        # media_list = (await gopro.http_command.get_media_list()).data.files
        # # for item in media_list:
        # #     print(item)
        # await gopro.http_command.download_file(camera_file=media_list[-1].filename)
        
        print("GoPro process exited!")

def execute_main(commandqueue):
    print("GoPro process started!")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    loop.run_until_complete(main(commandqueue))
    loop.close()
    
    # asyncio.run(main())
    