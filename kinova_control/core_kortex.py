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

import time
import threading
import socket
import ast

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient

from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import ControlConfig_pb2

import kinova_control.utilities as utilities

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20


def mediator(interface_port, mediator_port, kinova_ip, username, password):
    """
    A mediator between the control interface and core of Kinova Gen3 so that multiple robots can run parallelly.
    :param interface_port: the UDP port for the control interface
    :param mediator_port: the UDP port for this mediator
    :param kinova_ip: the IP address of the Kinova Gen3 robot to connect
    :param username: the username for connection
    :param password: the password for connection
    """

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('', mediator_port))

    connection_args = (kinova_ip, username, password)
    with utilities.DeviceConnection.createTcpConnection(connection_args) as router:
        control_core = KinovaGen3Core(router)
        udp_socket.sendto(b'Ready', ('', interface_port))

        while True:
            command_str, addr = udp_socket.recvfrom(256)
            command = command_str.decode().split(';')

            if command[0] == 'disconnect':
                break

            if command[0] == 'wait':
                udp_socket.sendto(b'1', ('', interface_port))
                continue

            result = control_core.execute(command)

            if command[-1] == '1':
                udp_socket.sendto(str(result).encode(), ('', interface_port))


def check_for_end_or_abort(e, verbose=False):
    """
    Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification):
        if verbose:
            if notification.action_event == 11:
                print("EVENT : ACTION_FEEDBACK")
            else:
                print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))

        if notification.action_event == Base_pb2.ACTION_END \
                or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()

    return check


class KinovaGen3Core:
    """
    The remote-control core of Kinova Gen3.
    """

    def __init__(self, router, verbose=False):
        """
        Initialization. Creates connection to the device and gets the router.
        :param router: utilities.DeviceConnection to connect to the robot
        :param verbose: True if printing log, False otherwise
        """

        self.base = BaseClient(router)
        self.base_cyclic = BaseCyclicClient(router)
        self.control_config_client = ControlConfigClient(router)
        self.verbose = verbose

        # # Sets the Cartesian reference frame
        # self.set_cartesian_reference_frame("", ControlConfig_pb2.CARTESIAN_REFERENCE_FRAME_TOOL)

    def execute(self, command):
        """
        Executes the given command.
        :param command: an iterable that contains a method name and the corresponding arguments.
        :return: the return value from the method executed
        """

        method = getattr(self, command[0])

        return method() if len(command) == 2 else method(*command)

    def get_base_data(self):
        """
        Gets the base data.
        :return: the base data
        """

        return self.base_cyclic.RefreshFeedback().base

    def get_cartesian_pose_quat(self):
        """
        Gets the cartesian pose.
        :return: the cartesian pose [x, y, z, theta_x, theta_y, theta_z]
        """

        return []

    def get_cartesian_pose_euler(self):
        """
        Gets the cartesian pose.
        :return: the cartesian pose [x, y, z, theta_x, theta_y, theta_z]
        """

        base = self.get_base_data()
        return [base.tool_pose_x, base.tool_pose_y, base.tool_pose_z,
                base.tool_pose_theta_x, base.tool_pose_theta_y, base.tool_pose_theta_z]

    def get_tool_forces_torques(self):
        """
        Gets the tool forces and torques.
        :return: the tool forces [Fx, Fy, Fz, Mx, My, Mz]
        """

        base = self.get_base_data()
        return [base.tool_external_wrench_force_x,
                base.tool_external_wrench_force_y,
                base.tool_external_wrench_force_z,
                base.tool_external_wrench_torque_x,
                base.tool_external_wrench_torque_y,
                base.tool_external_wrench_torque_z]

    def stop(self):
        """
        Stops the robot.
        :return: True always
        """

        if self.verbose:
            print("Stopping the robot...")
        self.base.Stop()
        time.sleep(1)

        return True

    def open_gripper(self):
        """
        Opens the gripper.
        :return: True if succeeds, False otherwise
        """

        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Set speed to open gripper
        if self.verbose:
            print("Opening gripper using speed command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = 0.1
        self.base.SendGripperCommand(gripper_command)

        # Wait for reported position to be opened
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len(gripper_measure.finger):
                if self.verbose:
                    print("Current position is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value < 0.01:
                    return True
            else:  # Else, no finger present in answer, end loop
                return False

    def close_gripper(self):
        """
        Closes the gripper.
        :return: True if succeeds, False otherwise
        """

        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Set speed to close gripper
        if self.verbose:
            print("Closing gripper using speed command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = -0.1
        self.base.SendGripperCommand(gripper_command)

        # Wait for reported speed to be 0
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len(gripper_measure.finger):
                if self.verbose:
                    print("Current speed is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value == 0.0:
                    return True
            else:  # Else, no finger present in answer, end loop
                return False

    def set_cartesian_reference_frame(self, *args):
        reference_frame = args[1]

        cartesian_reference_frame_info = ControlConfig_pb2.CartesianReferenceFrameInfo()
        cartesian_reference_frame_info.reference_frame = reference_frame
        self.control_config_client.SetCartesianReferenceFrame(cartesian_reference_frame_info)

        time.sleep(0.25)
        return True

    def reach_defined_position(self, *args):
        """
        Moves the robot to a defined position.
        :param args: defined_position: Home or Retract
        :return: True if succeeds, False otherwise
        """

        defined_position = args[1]

        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Move arm to the home position
        if self.verbose:
            print("Moving the arm to the defined position")

        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name.lower() == defined_position.lower():
                action_handle = action.handle
                break

        if action_handle is None:
            if self.verbose:
                print("Can't reach the home position. Exiting")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e, verbose=self.verbose),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            if self.verbose:
                print("Safe position reached")
        else:
            if self.verbose:
                print("Timeout on action notification wait")

        return finished

    def reach_joint_angles(self, *args):
        """
        Moves the robot's joints to the target angles.
        :param args: target_angles: an iterable that contains 7 int for degrees
        :return: True if succeeds, False otherwise
        """

        target_angles = ast.literal_eval(args[1])

        if self.verbose:
            print("Starting angular action movement ...")

        action = Base_pb2.Action()
        action.name = "Angular action movement"
        action.application_data = ""

        actuator_count = self.base.GetActuatorCount()

        # Place arm straight up
        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = target_angles[joint_id]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e, verbose=self.verbose),
            Base_pb2.NotificationOptions()
        )

        if self.verbose:
            print("Executing action")
        self.base.ExecuteAction(action)

        if self.verbose:
            print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            if self.verbose:
                print("Angular movement completed")
        else:
            if self.verbose:
                print("Timeout on action notification wait")

        return finished

    def reach_cartesian_pose(self, *args):
        """
        Moves the robot to the target pose.
        :param args: target_pose: an iterable that contains 6 float for [x, y, z, theta_x, theta_y, theta_z]
        :return: True if succeeds, False otherwise
        """

        target_pose = ast.literal_eval(args[1])

        if self.verbose:
            print("Starting Cartesian action movement ...")

        action = Base_pb2.Action()
        action.name = "Cartesian action movement"
        action.application_data = ""

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = target_pose[0]  # meter
        cartesian_pose.y = target_pose[1]  # meter
        cartesian_pose.z = target_pose[2]  # meter
        cartesian_pose.theta_x = target_pose[3]  # degree
        cartesian_pose.theta_y = target_pose[4]  # degree
        cartesian_pose.theta_z = target_pose[5]  # degree

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e, verbose=self.verbose),
            Base_pb2.NotificationOptions()
        )

        if self.verbose:
            print("Executing action")
        self.base.ExecuteAction(action)

        if self.verbose:
            print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            if self.verbose:
                print("Cartesian movement completed")
        else:
            if self.verbose:
                print("Timeout on action notification wait")

        return finished

    def twist(self, *args):
        """
        Moves the robot to the target pose.
        :param args: target_twist: an iterable that contains 6 float
                                   for [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]
                     duration: the duration of movement; the movement will not stop when duration = 0
        :return: True if succeeds, False otherwise
        """

        target_twist = ast.literal_eval(args[1])
        duration = 5 if len(args) == 3 else int(args[2])

        command = Base_pb2.TwistCommand()

        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
        command.duration = 0

        twist = command.twist
        twist.linear_x = target_twist[0]  # meter / second
        twist.linear_y = target_twist[1]  # meter / second
        twist.linear_z = target_twist[2]  # meter / second
        twist.angular_x = target_twist[3]  # degree / second
        twist.angular_y = target_twist[4]  # degree / second
        twist.angular_z = target_twist[5]  # degree / second

        self.base.SendTwistCommand(command)

        # Let time for twist to be executed
        if duration > 0:
            if self.verbose:
                print("Sending the twist command for 5 seconds...")
            time.sleep(duration)
            self.stop()

        return True

    def reach_gripper_position(self, *args):
        """
        Sets the gripper's position.
        :param args: target_position: a float for target_position
        :return: True always
        """

        target_position = ast.literal_eval(args[1])

        if target_position < 0.0 or target_position > 1.0:
            return False

        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        if self.verbose:
            print("Performing gripper test in position...")
        gripper_command.mode = Base_pb2.GRIPPER_POSITION

        finger.finger_identifier = 1
        finger.value = target_position
        if self.verbose:
            print("Going to position {:0.2f}...".format(finger.value))
        self.base.SendGripperCommand(gripper_command)

        # Wait for reported speed to be 0
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len(gripper_measure.finger):
                if self.verbose:
                    print("Current speed is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value == 0.0:
                    return True
            else:  # Else, no finger present in answer, end loop
                return False

    def reach_cartesian_waypoint(self, *args):
        """
        Moves the robot to the target pose.
        :param args: target_pose: an iterable that contains 6 float for [x, y, z, theta_x, theta_y, theta_z]
        :return: True if succeeds, False otherwise
        """

        target_pose = ast.literal_eval(args[1])
        velocity = float(args[2])

        if self.verbose:
            print("Starting Cartesian action movement ...")

        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        waypoints = Base_pb2.WaypointList()
        waypoints.duration = 0.0
        waypoints.use_optimal_blending = False

        waypoint = waypoints.waypoints.add()
        waypoint.name = "Cartesian trajectory movement"
        waypoint.cartesian_waypoint.pose.x = target_pose[0]
        waypoint.cartesian_waypoint.pose.y = target_pose[1]
        waypoint.cartesian_waypoint.pose.z = target_pose[2]
        waypoint.cartesian_waypoint.blending_radius = 0.0
        waypoint.cartesian_waypoint.pose.theta_x = target_pose[3]
        waypoint.cartesian_waypoint.pose.theta_y = target_pose[4]
        waypoint.cartesian_waypoint.pose.theta_z = target_pose[5]
        waypoint.cartesian_waypoint.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        waypoint.cartesian_waypoint.maximum_linear_velocity = velocity

        # Verify validity of waypoints
        result = self.base.ValidateWaypointList(waypoints)
        if len(result.trajectory_error_report.trajectory_error_elements) == 0:
            e = threading.Event()
            notification_handle = self.base.OnNotificationActionTopic(check_for_end_or_abort(e),
                                                                      Base_pb2.NotificationOptions())

            if self.verbose:
                print("Moving cartesian trajectory...")

            self.base.ExecuteWaypointTrajectory(waypoints)

            if self.verbose:
                print("Waiting for trajectory to finish ...")
            finished = e.wait(TIMEOUT_DURATION)
            self.base.Unsubscribe(notification_handle)

            if finished:
                if self.verbose:
                    print("Cartesian trajectory with no optimization completed ")
                e_opt = threading.Event()
                notification_handle_opt = self.base.OnNotificationActionTopic(check_for_end_or_abort(e_opt),
                                                                              Base_pb2.NotificationOptions())

                waypoints.use_optimal_blending = True
                self.base.ExecuteWaypointTrajectory(waypoints)

                if self.verbose:
                    print("Waiting for trajectory to finish ...")
                finished_opt = e_opt.wait(TIMEOUT_DURATION)
                self.base.Unsubscribe(notification_handle_opt)

                if self.verbose:
                    if finished_opt:
                        print("Cartesian trajectory with optimization completed ")
                    else:
                        print("Timeout on action notification wait for optimized trajectory")

                return finished_opt
            else:
                if self.verbose:
                    print("Timeout on action notification wait for non-optimized trajectory")

            return finished

        else:
            if self.verbose:
                print("Error found in trajectory")
                print(result.trajectory_error_report.trajectory_error_elements)
            # result.trajectory_error_report.PrintDebugString()
