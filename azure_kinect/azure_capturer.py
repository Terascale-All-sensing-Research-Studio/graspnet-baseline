#!/usr/bin/env python

import os
import time

import numpy as np
import open3d as o3d
from PIL import Image

if __name__ == "__main__":
    config = o3d.io.read_azure_kinect_sensor_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "kinect_config.json"))
    config = o3d.io.read_azure_kinect_sensor_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                 "kinect_config_photo.json"))
    sensor = o3d.io.AzureKinectSensor(config)

    assert sensor.connect(0)

    time.sleep(1)
    rgbd = None
    while rgbd is None:
        # Get rgbd image
        rgbd = sensor.capture_frame(True)

    sensor.disconnect()

    color = np.array(rgbd.color).astype(np.uint8)
    depth = np.array(rgbd.depth).astype(np.uint16)

    assert depth.shape[0] != 0
    assert depth.shape[:2] == color.shape[:2]

    Image.fromarray(color).save("kinect_test_rgb.png")
    Image.fromarray(depth).save("kinect_test_depth.png")
