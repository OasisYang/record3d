import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
import os
import json
import argparse
import shutil
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='relocate', help='task name')
parser.add_argument('--object', type=str, default='sugar', help='object name')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--start_over', action='store_true')

class_names = [
    '002_master_chef_can',
    '003_cracker_box',
    '004_sugar_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '009_gelatin_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '036_wood_block',
    '037_scissors',
    '040_large_marker',
    '051_large_clamp',
    '052_extra_large_clamp',
    '061_foam_brick'
]


class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])

    def start_processing_stream(self, task, object, resume=False):
        data_root = os.path.join('raw_data', task, opt.object)
        os.makedirs(data_root, exist_ok=True)
        folders = sorted(os.listdir(data_root))
        if len(folders) == 0:
            cur_folder = '{:06d}'.format(0)
        else:
            pre_folder = folders[-1]
            if resume:
                cur_folder = pre_folder
                shutil.rmtree(os.path.join(data_root, cur_folder))
            else:
                cur_folder = '{:06d}'.format(int(pre_folder)+1)

        vid_root = os.path.join(data_root, cur_folder)
        print("Number of video:{}".format(int(cur_folder)+1))
        os.makedirs(vid_root, exist_ok=True)
        cur_frame = 0
        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())

            # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.
            meta = {}
            meta['H'] = 640
            meta['W'] = 480
            meta['K'] = intrinsic_mat.tolist()
            with open(os.path.join(vid_root, 'meta.json'), 'w') as f:
                json.dump(meta, f)

            # Postprocess it
            # if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
            #     depth = cv2.flip(depth, 1)
            #     rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb_n = os.path.join(vid_root, '{:06d}-color.png'.format(cur_frame))
            depth_n = rgb_n.replace('color', 'depth')
            # pdb.set_trace()
            cv2.imwrite(rgb_n, rgb)
            cv2.imwrite(depth_n, (1000*depth).astype(np.uint16))
            # Show the RGBD Stream
            cv2.imshow('RGB', rgb)
            cv2.imshow('Depth', depth)
            cv2.waitKey(1)
            cur_frame += 1
            self.event.clear()


if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.start_over:
        shutil.rmtree(os.path.join('raw_data', opt.task, opt.object))
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream(task=opt.task, resume=opt.resume)
