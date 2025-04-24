#!/usr/bin/env python

import socket
import pickle

from PIL import Image


def receive_numpy_array(sock):
    size_data = sock.recv(4)
    size = int.from_bytes(size_data, 'big')
    data = b''

    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet

    return pickle.loads(data)


def request_image(sock, index):
    sock.sendall(index.to_bytes(4, 'big'))
    return receive_numpy_array(sock)


def capture(index):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.0.119', 12345))

    index_mapping = {0: 2, 1: 0, 2: 3, 3: 1}
    images = request_image(client_socket, index)
    if images is not None:
        print("Received image")
        Image.fromarray(images['color']).save("kinect_test_%d_rgb.png" % index_mapping[index])
        Image.fromarray(images['depth']).save("kinect_test_%d_depth.png" % index_mapping[index])
    else:
        print("Failed to receive image")

    client_socket.close()


def capture_all():
    for i in range(4):
        capture(i)


if __name__ == '__main__':
    capture_all()
