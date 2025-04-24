import numpy as np

gripper_max_width = 0.14
gripper_link_length = 0.1
grippe_open_total_height = 0.206
gripper_open_link_angle = 49.50
gripper_open_link_height = gripper_link_length * np.sin(np.radians(gripper_open_link_angle))
gripper_open_link_width = gripper_link_length * np.cos(np.radians(gripper_open_link_angle))
gripper_link_close_angle = 93.06
gripper_close_link_height = gripper_link_length * np.sin(np.radians(gripper_link_close_angle))
gripper_close_link_width = gripper_link_length * np.cos(np.radians(gripper_link_close_angle))


def get_gripper_absolute_width(relative_width_ratio):
    return relative_width_ratio * gripper_max_width


def get_gripper_offset(width):
    if width < 0.0:
        width = 0.0
    elif width > gripper_max_width:
        width = gripper_max_width

    height = (np.sqrt(gripper_link_length ** 2 - (0.5 * width + gripper_close_link_width) ** 2)
              - gripper_open_link_height + grippe_open_total_height)
    return height


if __name__ == "__main__":
    print(get_gripper_offset(0.0))
    print(get_gripper_offset(0.07))
    print(get_gripper_offset(0.14))
