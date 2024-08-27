# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/azure_kinect_recorder.py

import argparse
import datetime
import open3d as o3d
import os
import time

class RecorderWithCallback:

    def __init__(self, config, device, filename, align_depth_to_color):
        self.flag_exit = False
        self.flag_record = False
        self.filename = filename
        self.rootdir = ""
        self.align_depth_to_color = align_depth_to_color
        self.config = o3d.io.read_azure_kinect_sensor_config('D:/repos/EMS-Pipeline/DataCollection/Kinect/config.json')
        self.recorder = o3d.io.AzureKinectRecorder(self.config, device)
        if not self.recorder.init_sensor():
            raise RuntimeError('Failed to connect to sensor')
        self.vis = None
        self.ts_file = None


    def start_recording(self, root):
        if not self.recorder.is_record_created():
            recording_path = f"{root}/Kinect"
           # create dir if not exists
            if not os.path.exists(recording_path):
                os.makedirs(recording_path)

            self.rootdir = recording_path
                
            recording_path = f"{recording_path}/{self.filename}"
            print('filename kinect: ' ,recording_path)

            # open ts file
            ts_filename = f"{self.rootdir}/ts.txt"
            self.ts_file = open(ts_filename, 'a')
            
            if self.recorder.open_record(recording_path):
                print('Recording started.')
                self.flag_record = True
        else:
            print('Recording resumed, video may be discontinuous.')
            self.flag_record = True

    def pause_recording(self):
        if self.flag_record:
            print('Recording paused.')
            self.flag_record = False

    def stop_recording(self):
        self.flag_exit = True
        if self.recorder.is_record_created():
            print('Recording finished.')
        else:
            print('Nothing has been recorded.')
        # self.recorder.close_record()
        return False

    def get_ts(self):
        t=str(time.time_ns())
        return t


    def run(self):
        glfw_key_escape = 256
        glfw_key_space = 32
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(glfw_key_escape, self.escape_callback)
        self.vis.register_key_callback(glfw_key_space, self.space_callback)

        self.vis.create_window('recorder', 1920, 540)
        print("Recorder initialized. Press [SPACE] to start. "
              "Press [ESC] to save and exit.")

        vis_geometry_added = False
        while not self.flag_exit:
            rgbd = self.recorder.record_frame(self.flag_record,
                                              self.align_depth_to_color)
            if rgbd is None:
                continue

                #write ts to the ts file
            ts=self.get_ts()
            if(self.ts_file):
                self.ts_file.write(ts + "\n")
                self.ts_file.flush()
      
            if not vis_geometry_added:
                self.vis.add_geometry(rgbd)
                vis_geometry_added = True

            self.vis.update_geometry(rgbd)
            self.vis.poll_events()
            self.vis.update_renderer()

        self.recorder.close_record()

    def escape_callback(self, vis):
        self.stop_recording()
        return False

    def space_callback(self, vis):
        if self.flag_record:
            self.pause_recording()
        else:
            self.start_recording()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect mkv recorder.')
    parser.add_argument('--config', type=str, help='input json kinect config')
    parser.add_argument('--output', type=str, help='output mkv filename')
    parser.add_argument('--list',
                        action='store_true',
                        help='list available azure kinect sensors')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='input kinect device id')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()

    if args.list:
        o3d.io.AzureKinectSensor.list_devices()
        exit()

    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    if args.output is not None:
        filename = args.output
    else:
        filename = '{date:%Y-%m-%d-%H-%M-%S}.mkv'.format(
            date=datetime.datetime.now())
    print('Prepare writing to {}'.format(filename))

    device = args.device
    if device < 0 or device > 255:
        print('Unsupported device id, fall back to 0')
        device = 0

    r = RecorderWithCallback(config, device, filename,
                             args.align_depth_to_color)
    
    r.run()