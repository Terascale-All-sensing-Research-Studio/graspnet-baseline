import os
import sys
import argparse
import time
import pickle
import json
import random

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from PIL import Image
import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'kinova_control'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from interface import KinovaGen3

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01,
                    help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
cfgs = parser.parse_args()

random.seed(cfgs.seed)
np.random.seed(cfgs.seed)
torch.manual_seed(cfgs.seed)
torch.cuda.manual_seed_all(cfgs.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def clean_point_cloud(points, colors, k=30, threshold=0.001):
    tree = KDTree(points)

    # Calculate the distances to the k nearest neighbors for each point
    distances, _ = tree.query(points, k=k)

    pts_removed_by_dist = np.vstack((
        np.linalg.norm(points, axis=1) < 0.2,  # if points are too close to the cam
        distances[:, -1] > threshold,  # if neighbors are too far away
    )).any(axis=0)

    # Filter out points that have mean distances above the threshold
    cleaned_points = np.delete(points, pts_removed_by_dist, axis=0)

    # If color data is present, filter the color data as well
    cleaned_colors = np.delete(colors, pts_removed_by_dist, axis=0)

    return cleaned_points, cleaned_colors


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


def capture(capture_name="capture"):
    config = o3d.io.read_azure_kinect_sensor_config("./azure_kinect/azure_config.json")
    sensor = o3d.io.AzureKinectSensor(config)

    assert sensor.connect(1)

    time.sleep(2)
    rgbd = None
    while rgbd is None:
        # Get rgbd image
        rgbd = sensor.capture_frame(True)

    sensor.disconnect()

    color = np.array(rgbd.color).astype(np.uint8)
    depth = np.array(rgbd.depth).astype(np.uint16)

    assert depth.shape[0] != 0
    assert depth.shape[:2] == color.shape[:2]

    Image.fromarray(color).save(os.path.join(experiment_name, f"{experiment_name}_{capture_name}_rgb.png"))
    Image.fromarray(depth).save(os.path.join(experiment_name, f"{experiment_name}_{capture_name}_depth.png"))

    return color, depth


def get_and_process_data():
    # load data
    azure_params = json.load(open("./azure_kinect/azure_params.json", "r"))
    intrinsic = np.array(azure_params['intrinsic_matrix'])
    factor_depth = 1000.0
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    color0, depth0 = capture()
    color0 = color0.astype(np.float32) / 255.0
    points0 = create_point_cloud_from_depth_image(depth0, camera, organized=True)
    points0 = points0.reshape(-1, 3)
    color0 = color0.reshape(-1, 3)

    cloud_points = points0.astype(np.float32)
    cloud_colors = color0.astype(np.float32)
    mask = (((cloud_points[:, 2] < 0.55) & (cloud_points[:, 2] > 0.1))
            & ((cloud_points[:, 0] < 0.25) & (cloud_points[:, 0] > -0.25))
            & ((cloud_points[:, 1] < 0.25) & (cloud_points[:, 1] > -0.25)))
    cloud_masked = cloud_points[mask]
    color_masked = cloud_colors[mask]
    cloud_masked, color_masked = clean_point_cloud(cloud_masked, color_masked, k=30, threshold=0.01)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    end_points = dict()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg


def select_grasp(robot, gg, robot_current_pose):
    gg.nms()
    gg.sort_by_score()

    klampt_ik = robot.klampt_ik
    for g in gg:
        g_translation = g.translation
        g_rotation = g.rotation_matrix
        robot_translation, robot_orientation = calculate_robot_pose(g_translation,
                                                                    g_rotation,
                                                                    g.depth,
                                                                    robot_current_pose)
        reachable = klampt_ik.solve_ik(list(robot_translation), list(robot_orientation))[0]
        if reachable:
            g_selected = g
            translation_selected = robot_translation
            orientation_selected = robot_orientation
            break
    else:
        raise ValueError("No good grasp found.")

    print("Original translation:", g_selected.translation)
    print("Original Euler angles: ", R.from_matrix(g_selected.rotation_matrix).as_euler('xyz', degrees=True))
    print("Original width:", g_selected.width)

    pose_selected = np.concatenate((translation_selected, np.rad2deg(orientation_selected)), axis=0)
    pose_selected = np.round(pose_selected, 4)
    print("Robot target pose:", list(pose_selected))

    return g_selected, pose_selected


def vis_grasps(gg, cloud):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    gripper = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([coordinate_frame, cloud, *gripper])


def vis_grasp(g, cloud):
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    gripper = g.to_open3d_geometry()
    centroid_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.0075)
    centroid_mesh.translate(g.translation)
    centroid_mesh.paint_uniform_color([1, 0, 1])
    o3d.visualization.draw_geometries([coordinate_frame, cloud, gripper, centroid_mesh])


def calculate_robot_pose(g_translation, g_rotation, depth, robot_current_pose):
    # Rotation matrices
    R1 = R.from_euler('x', 180, degrees=True)
    R2 = R.from_euler('z', 90, degrees=True)
    R3 = R.from_euler('z', 180, degrees=True)
    R4 = R.from_euler('x', 90, degrees=True)
    R5 = R.from_euler('z', 90, degrees=True)

    # Translation
    T = np.eye(4)
    T[:3, :3] = g_rotation
    _depth = np.array([depth, 0, 0, 1])
    offset = np.dot(T, _depth).flatten()
    g_translation_adjusted = np.array([[g_translation[0] + offset[0], ],
                                       [g_translation[1] + offset[1], ],
                                       [g_translation[2] + offset[2], ],
                                       [1, ]])
    T = np.eye(4)
    T[:3, :3] = (R1 * R2).as_matrix()
    T[:3, 3] = np.array([robot_current_pose[0] + azure_offset_y,
                         robot_current_pose[1] - azure_offset_x,
                         robot_current_pose[2] + azure_offset_z])
    robot_translation = np.dot(T, g_translation_adjusted).flatten()[:3]

    # Rotation
    robot_rotation = (R1 * R2) * (R.from_matrix(g_rotation) * R5 * R4 * R3)
    robot_orientation = robot_rotation.as_euler('xyz')

    return robot_translation, robot_orientation


def main():
    # Connect to the robot
    robot = KinovaGen3(kinova_ip="192.168.1.12", use_ros=False)
    robot.connect()

    # Set robot to initial position
    robot.open_gripper()
    robot.reach_joint_angles(robot_start_angles)
    robot.reach_cartesian_waypoint(robot_start_pose, velocity=0.1)

    # Ensure robot is at the initial position
    robot_current_pose = robot.get_cartesian_pose()
    if abs(robot_current_pose[0] - robot_x_initial) > 0.01:
        raise ValueError("Robot is not at the initial position.")

    # Load the model
    net = get_net()

    # Capture and process data
    end_points, cloud = get_and_process_data()

    # Get all grasps
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    # vis_grasps(gg, cloud)

    # Select a grasp
    g_selected, target_pose = select_grasp(robot, gg, robot_current_pose)
    vis_grasp(g_selected, cloud)

    # Execute the grasp
    robot.reach_cartesian_waypoint(target_pose, velocity=0.1)
    robot.close_gripper()
    time.sleep(2)

    # Move to the end position
    robot.reach_cartesian_waypoint(robot_end_pose, velocity=0.1)

    # Ensure robot is at the end position
    robot_current_pose = robot.get_cartesian_pose()
    if abs(robot_current_pose[2] - robot_z_end) > 0.01:
        robot.reach_joint_angles(robot_end_angles)

    # Save the data
    data_saved = {
        "gg": gg,
        "gg_selected": g_selected,
        "robot_pose": target_pose
    }
    with open(os.path.join(experiment_dir, f'{experiment_name}.pkl'), 'wb') as f:
        pickle.dump(data_saved, f)

    # Disconnect from the robot
    robot.disconnect()


if __name__ == '__main__':
    # Robot initial and end poses
    robot_x_initial, robot_y_initial, robot_z_initial = 0.5, 0, 0.34
    robot_x_end, robot_y_end, robot_z_end = 0.5, 0, 0.24
    robot_start_pose = (robot_x_initial, robot_y_initial, robot_z_initial, 180, 0, 90)
    robot_start_angles = (5, 27, 171, 306, 4, 262, 88)  # Corresponding to the robot_start_pose
    robot_end_pose = robot_start_pose
    robot_end_angles = robot_start_angles

    # Azure Kinect offsets and camera height
    azure_offset_x, azure_offset_y, azure_offset_z = 0.024, 0.1, 0.14
    camera_z = robot_z_initial + azure_offset_z

    # Create the experiment directory
    experiment_name = time.strftime('%Y%m%d%H%M%S', time.localtime())
    experiment_dir = os.path.join("..", experiment_name)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    main()
