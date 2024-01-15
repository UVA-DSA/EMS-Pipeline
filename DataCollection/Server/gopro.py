import asyncio
from open_gopro import WirelessGoPro, Params
import os
import shutil


def move_files_with_prefix(src_folder, dest_folder, prefix):
    # Ensure the source and destination folders exist
    if not os.path.exists(src_folder):
        print(f"Source folder '{src_folder}' does not exist.")
        return
    
    if not os.path.exists(dest_folder):
        print(f"Destination folder '{dest_folder}' does not exist.")
        return

    # Iterate through files in the source folder
    for filename in os.listdir(src_folder):
        if filename.startswith(prefix):
            # Construct the full paths for source and destination
            src_path = os.path.join(src_folder, filename)
            dest_path = os.path.join(dest_folder, filename)

            # Move the file
            shutil.move(src_path, dest_path)
            print(f"Moved '{filename}' to '{dest_folder}'.")

async def main(commandqueue, path):
    async with WirelessGoPro() as gopro:
        
        print("GoPro Directory: ",path)
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
                
                print("Download complete! Moving file..")
                move_files_with_prefix("./",path,"GH")

            
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

def execute_main(commandqueue,path):
    print("GoPro process started!")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    loop.run_until_complete(main(commandqueue,path))
    loop.close()
    
    # asyncio.run(main())
    