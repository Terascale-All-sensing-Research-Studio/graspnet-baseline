#!/usr/bin/env python

import os
import time
import socket
import pickle

import numpy as np
import open3d as o3d

config = o3d.io.read_azure_kinect_sensor_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                             "kinect_config.json"))
sensor = o3d.io.AzureKinectSensor(config)


def send_numpy_array(sock, images):
    data = pickle.dumps(images)
    sock.sendall(len(data).to_bytes(4, 'big'))
    sock.sendall(data)


def capture(idx):
    if not sensor.connect(idx):
        return None, None

    time.sleep(1)
    rgbd = None
    while rgbd is None:
        # Get rgbd image
        rgbd = sensor.capture_frame(True)

    sensor.disconnect()

    color = np.array(rgbd.color).astype(np.uint8)
    depth = np.array(rgbd.depth).astype(np.uint16)

    if depth.shape[0] == 0 or depth.shape[:2] != color.shape[:2]:
        return None, None

    return color, depth


def main():
    # Socket setup
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 12345
    server_socket.bind(('192.168.0.119', port))
    server_socket.listen(5)
    print("Listening at:", port)

    while True:
        # Accept a client connection
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")

        try:
            index_data = client_socket.recv(4)
            if not index_data:
                continue

            index = int.from_bytes(index_data, 'big')
            if 0 <= index < 4:
                color, depth = capture(index)
                images = {'color': color, 'depth': depth}
                send_numpy_array(client_socket, images)
            else:
                print(f"Invalid index received: {index}")

        finally:
            client_socket.close()


if __name__ == "__main__":
    main()
