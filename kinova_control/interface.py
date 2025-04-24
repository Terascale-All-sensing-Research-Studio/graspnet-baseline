###
#
# Created by Xinchao Song, 2022/06/22
# https://github.com/xinchaosong
#
###

import multiprocessing
import socket
import ast
import subprocess
from pathlib import Path

import numpy as np

from kinova_control.core_kortex import mediator


class KinovaGen3:
    """
    The interface of Kinova Gen3. Wraps the remote-control core (core.KinovaGen3Core) and supports multiple robots
    through the mediator function using multiprocessing.
    """

    def __init__(self, kinova_ip="192.168.1.10", kinova_name="gen3", username="admin", password="admin", use_ros=False):
        """
        Initialization.

        :param kinova_ip: the IP address of the Kinova Gen3 robot to connect
        :param kinova_name: the name the Kinova Gen3 robot to connect
        :param username: the username for connection
        :param password: the password for connection
        """

        self.kinova_ip = kinova_ip
        self.kinova_name = kinova_name
        self.username = username
        self.password = password
        self.use_ros = use_ros

        port_offset = int(kinova_ip[kinova_ip.rfind('.') + 1:])
        if self.use_ros:
            port_offset += 1
        self.interface_port = 34300 + port_offset
        self.mediator_port = self.interface_port + 1

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.settimeout(600)
        self.udp_socket.bind(('', self.interface_port))
        self.control_process = None

    def connect(self):
        """
        Connects to the robot through the mediator.
        :return:
        """

        if self.use_ros:
            control_core_path = Path(__file__).resolve().parent / "core_ros.bash"
            subprocess.Popen(args=[str(control_core_path),
                                   str(self.interface_port),
                                   str(self.mediator_port),
                                   self.kinova_name])

        else:
            self.control_process = multiprocessing.Process(target=mediator,
                                                           args=(self.interface_port,
                                                                 self.mediator_port,
                                                                 self.kinova_ip,
                                                                 self.username,
                                                                 self.password))
            self.control_process.daemon = True
            self.control_process.start()

        self.udp_socket.recvfrom(16)
        print("Successfully connected to: ", self.kinova_ip)

    def disconnect(self):
        """
        Disconnects from the robot and the mediator.
        """

        self.udp_socket.sendto(b"disconnect", ('', self.mediator_port))
        if not self.use_ros:
            self.control_process.join()
        self.udp_socket.close()

    def wait(self):
        """
        Waits for the robot's action to be done.
        :return:
        """

        self.udp_socket.sendto(b"wait", ('', self.mediator_port))
        self.udp_socket.recvfrom(256)

    def get_cartesian_pose(self, quat=False):
        """
        Gets the cartesian pose.
        :param quat: True to get quaternion, False otherwise
        :return: the cartesian pose [x, y, z, theta_x, theta_y, theta_z] or [x, y, z, qx, qy, qz, qw]
        """

        if quat:
            return self._communicate("get_cartesian_pose_quat;1")
        else:
            return self._communicate("get_cartesian_pose_euler;1")

    def get_tool_forces_torques(self):
        """
        Gets the tool forces and torques.
        :return: the tool forces [Fx, Fy, Fz, Mx, My, Mz]
        """

        return self._communicate("get_tool_forces_torques;1")

    def reach_home_position(self, wait=True):
        """
        Moves the robot back to the home position.
        :param wait: True if waiting for the robot's action to be done, False otherwise
        :return: True if succeeds, False otherwise
        """

        return self._communicate("reach_defined_position;home;%d" % wait)

    def reach_retract_position(self, wait=True):
        """
        Moves the robot back to the home position.
        :param wait: True if waiting for the robot's action to be done, False otherwise
        :return: True if succeeds, False otherwise
        """

        return self._communicate("reach_defined_position;retract;%d" % wait)

    def stop(self):
        """
        Stops the robot.
        :return: True always
        """

        return self._communicate("stop;1")

    def reach_joint_angles(self, target_angles, wait=True):
        """
        Moves the robot's joints to the target angles.
        :param target_angles: an iterable that contains 7 int for degrees
        :param wait: True if waiting for the robot's action to be done, False otherwise
        :return: True if succeeds, False otherwise
        """

        return self._communicate("reach_joint_angles;%s;%d" % (target_angles, wait))

    def reach_cartesian_pose(self, target_pose, velocity=1, straight=False, wait=True):
        """
        Moves the robot to the target pose.
        :param target_pose: an iterable that contains 6 float for [x, y, z, theta_x, theta_y, theta_z]
        :param velocity: velocity (0, 1]
        :param straight: True if moving along a straight line
        :param wait: True if waiting for the robot's action to be done, False otherwise
        :return: True if succeeds, False otherwise
        """

        return self._communicate("reach_cartesian_pose;%s;%f;%d;%d"
                                 % (list(target_pose), float(velocity), straight, wait))

    def reach_cartesian_waypoint(self, target_pose, velocity=0.1, straight=False, wait=True):
        """
        Moves the robot to the target pose.
        :param target_pose: an iterable that contains 6 float for [x, y, z, theta_x, theta_y, theta_z]
        :param velocity: velocity (0, 1]
        :param straight: True if moving along a straight line
        :param wait: True if waiting for the robot's action to be done, False otherwise
        :return: True if succeeds, False otherwise
        """

        return self._communicate("reach_cartesian_waypoint;%s;%f;%d;%d"
                                 % (list(target_pose), float(velocity), straight, wait))

    def reach_tool_pose(self, target_pose, duration=5, wait=True):
        """
        Moves the robot to the target pose.
        :param target_pose: an iterable that contains 6 float for [x, y, z, theta_x, theta_y, theta_z]
        :param duration: the duration of movement; the movement will not stop when duration = 0
        :param wait: True if waiting for the robot's action to be done, False otherwise
        :return: True if succeeds, False otherwise
        """

        target_twist = list(np.array(target_pose) / duration)

        return self.twist(target_twist, duration=duration, wait=wait)

    def twist(self, target_twist, duration=5, wait=True):
        """
        Moves the robot to the target pose.
        :param target_twist: an iterable that contains 6 float
                             for [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
        :param duration: the duration of movement; the movement will not stop when duration = 0
        :param wait: True if waiting for the robot's action to be done, False otherwise
        :return: True if succeeds, False otherwise
        """

        return self._communicate("twist;%s;%d;%d" % (target_twist, duration, wait))

    def reach_gripper_position(self, target_position, wait=True):
        """
        Sets the gripper's position.
        :param target_position: a float for target_position
        :param wait: True if waiting for the robot's action to be done, False otherwise
        :return: True always
        """

        return self._communicate("reach_gripper_position;%f;%d" % (target_position, wait))

    def open_gripper(self):
        """
        Opens the gripper.
        :return: True if succeeds, False otherwise
        """
        return self._communicate("open_gripper;1")

    def close_gripper(self):
        """
        Closes the gripper.
        :return: True if succeeds, False otherwise
        """
        return self._communicate("close_gripper;1")

    def get_gripper_position(self):
        """
        Gets the gripper's position.
        :return: the gripper's position
        """
        return self._communicate("get_gripper_position;1")

    def _communicate(self, command):
        """
        Communicates with the mediator.
        :param command: the message string to send out
        :return: the response received
        """
        try:
            self.udp_socket.sendto(command.encode(), ('', self.mediator_port))

            if command[-1] == '1':
                data, addr = self.udp_socket.recvfrom(256)

                if data.decode() == "Error":
                    self.disconnect()
                    raise ConnectionError("The control core is crashed.")

                return ast.literal_eval(data.decode())

        except KeyboardInterrupt:
            self.disconnect()
