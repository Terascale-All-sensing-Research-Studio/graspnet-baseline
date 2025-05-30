###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###
#
# Modified by Xinchao Song, 2022/06/22
# https://github.com/xinchaosong
#
###

import argparse

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2


def parseConnectionArguments(parser=argparse.ArgumentParser()):
    parser.add_argument("--ip", type=str, help="IP address of destination", default="192.168.1.10")
    parser.add_argument("-u", "--username", type=str, help="username to login", default="admin")
    parser.add_argument("-p", "--password", type=str, help="password to login", default="admin")
    return parser.parse_args()


class DeviceConnection:
    TCP_PORT = 10000
    UDP_PORT = 10001

    @staticmethod
    def createTcpConnection(args):
        """
        returns RouterClient required to create services and send requests to device or sub-devices,
        """

        if isinstance(args, argparse.Namespace):
            ip = args.ip
            credentials = (args.username, args.password)
        else:
            ip = args[0]
            credentials = (args[1], args[2])

        return DeviceConnection(ip, port=DeviceConnection.TCP_PORT, credentials=credentials)

    @staticmethod
    def createUdpConnection(args):
        """
        returns RouterClient that allows to create services and send requests to a device or its sub-devices @ 1khz.
        """

        if isinstance(args, argparse.Namespace):
            ip = args.ip
            credentials = (args.username, args.password)
        else:
            ip = args[0]
            credentials = (args[1], args[2])

        return DeviceConnection(ip, port=DeviceConnection.UDP_PORT, credentials=credentials)

    def __init__(self, ipAddress, port=TCP_PORT, credentials=("", "")):

        self.ipAddress = ipAddress
        self.port = port
        self.credentials = credentials

        self.sessionManager = None

        # Setup API
        self.transport = TCPTransport() if port == DeviceConnection.TCP_PORT else UDPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    # Called when entering 'with' statement
    def __enter__(self):
        return self.open()

    # Called when exiting 'with' statement
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        self.transport.connect(self.ipAddress, self.port)

        if self.credentials[0] != "":
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000  # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)

            self.sessionManager = SessionManager(self.router)
            print("Logging as", self.credentials[0], "on device", self.ipAddress)
            self.sessionManager.CreateSession(session_info)

        return self.router

    def close(self):
        if self.sessionManager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000

            self.sessionManager.CloseSession(router_options)

        self.transport.disconnect()
