import math
import ast
import numpy as np
import random

HAND_IDENTIFIERS = [
    "wrist",
    "thumbCMC",
    "thumbMP",
    "thumbIP",
    "thumbTip",
    "indexMCP",
    "indexPIP",
    "indexDIP",
    "indexTip",
    "middleMCP",
    "middlePIP",
    "middleDIP",
    "middleTip",
    "ringMCP",
    "ringPIP",
    "ringDIP",
    "ringTip",
    "littleMCP",
    "littlePIP",
    "littleDIP",
    "littleTip"
]

IGNORE_COL_LIST = ['labels', 'video_size_width', 'video_fps', 'video_size_height']


def convert_data_to_python_rep(dataset, ignore_col_list=None):
    if ignore_col_list is None:
        ignore_col_list = []

    for row_index in range(len(dataset)):
        for column in dataset.columns:
            if column not in ignore_col_list:
                # print(column)
                data = ast.literal_eval(dataset.at[row_index, column])
                dataset.at[row_index, column] = data
    return dataset


def compute_angle_between(joint, joint_to_refer, extra_joint_to_refer=None):
    """
    Compute the angle between any given two or three vectors/joints
    """
    if not extra_joint_to_refer:
        # Calculate the dot product of the two vectors
        dot_product = sum(a * b for a, b in zip(joint, joint_to_refer))

        # Calculate the magnitudes of the vectors
        magnitude1 = math.sqrt(sum(a ** 2 for a in joint))
        magnitude2 = math.sqrt(sum(b ** 2 for b in joint_to_refer))

        # Calculate the cosine of the angle
        cosine_angle = dot_product / (magnitude1 * magnitude2)
        # Regulate the angle value to the range [-1, 1]
        cosine_angle = max(-1, min(1, cosine_angle))

        # Calculate the angle in radians
        angle_rad = math.acos(cosine_angle)

        # Convert angle to degrees
        angle_deg = math.degrees(angle_rad)

        # Determine the sign of the angle based on the cross product
        cross_product = joint[0] * joint_to_refer[1] - joint[1] * joint_to_refer[0]

        if cross_product < 0:
            # Angle is towards the left, so make it negative
            angle_deg = -angle_deg
    else:
        angle_rad = np.arctan2(extra_joint_to_refer[1] - joint_to_refer[1],
                               extra_joint_to_refer[0] - joint_to_refer[0]) \
                    - np.arctan2(joint[1] - joint_to_refer[1], joint[0] - joint_to_refer[0])
        angle_deg = math.degrees(angle_rad)  # angle_rad*180.0/np.pi

    return angle_deg


def get_angle_to_rotate(angle, valid_range):
    """
    angle: the angle to check
    valid_range: valid_range[0] is the minimum allowable number, valid_range[1] is the maximum allowable number
    """
    if valid_range[0] <= angle <= valid_range[1]:
        angle_to_rotate = 0
    elif angle > valid_range[1]:
        angle_to_rotate = valid_range[1] - angle
    else:  # angle < valid_range[0]
        angle_to_rotate = valid_range[0] - angle

    return angle_to_rotate


def get_angle_to_rotate_to_error(angle, valid_range, error_angle):
    """
    angle: the angle to check
    valid_range: valid_range[0] is the minimum allowable number, valid_range[1] is the maximum allowable number
    """
    opt = random.choice([0, 1])
    if int(opt) == 0:
        angle_to_rotate = (valid_range[0] - angle) - error_angle
    else:
        angle_to_rotate = (valid_range[1] - angle) + error_angle

    return angle_to_rotate


def rotate_joint(joint, angle_degrees, is_clock_wise=True):
    """
    Rotate a vector based on the given angle
    """
    # Convert angle to radians
    angle_radians = math.radians(angle_degrees)

    # Create the rotation matrix
    if is_clock_wise:
        rotation_matrix = [
            [math.cos(-angle_radians), -math.sin(-angle_radians)],
            [math.sin(-angle_radians), math.cos(-angle_radians)]
        ]
    else:
        rotation_matrix = [
            [math.cos(angle_radians), -math.sin(angle_radians)],
            [math.sin(angle_radians), math.cos(angle_radians)]
        ]

    # Perform the rotation using matrix multiplication
    rotated_joint = [
        rotation_matrix[0][0] * joint[0] + rotation_matrix[0][1] * joint[1],
        rotation_matrix[1][0] * joint[0] + rotation_matrix[1][1] * joint[1]
    ]

    return rotated_joint


def compute_statistics(num_frame_per_sample, num_sample, total_num_rectified_keypoints, total_num_rectified_hands,
                       rectified_frame, rectified_row):
    total_num_keypoints = 54 * num_frame_per_sample * num_sample
    total_num_hands = 2 * num_frame_per_sample * num_sample
    total_num_frame = num_frame_per_sample * num_sample
    # print(f"num_frame_per_sample {num_frame_per_sample}")  # 204
    # print(f"num_sample {num_sample}")  # 4000
    # print(f"total_num_keypoints {total_num_keypoints}")  # 44064000
    # print(f"total_num_hands {total_num_hands}")  # 1632000
    # print(f"total_num_frame {total_num_frame}")  # 816000

    rectified_keypoints_percentage = (total_num_rectified_keypoints / total_num_keypoints) * 100
    rectified_hands_percentage = (total_num_rectified_hands / total_num_hands) * 100
    rectified_frame = (len(rectified_frame) / total_num_frame) * 100
    rectified_data = (len(rectified_row) / num_sample) * 100

    print(f"rectified_keypoints_percentage = {rectified_keypoints_percentage}%, "
          f"rectified_hands_percentage = {rectified_hands_percentage}%, rectified_data = {rectified_data}% and "
          f"rectified_frame = {rectified_frame}%,"
          f"where total_num_keypoints = {total_num_keypoints}, "
          f"num_rectified_keypoints = {total_num_rectified_keypoints}")


def compute_simulation_statistics(num_frame_per_sample, num_sample, total_num_add_error_keypoints,
                                  total_num_add_error_hands,
                                  frame_error, row_error):
    total_num_keypoints = 54 * num_frame_per_sample * num_sample
    total_num_hands = 2 * num_frame_per_sample * num_sample
    total_num_frame = num_frame_per_sample * num_sample
    # print(f"num_frame_per_sample {num_frame_per_sample}")  # 204
    # print(f"num_sample {num_sample}")  # 4000
    # print(f"total_num_keypoints {total_num_keypoints}")  # 44064000
    # print(f"total_num_hands {total_num_hands}")  # 1632000
    # print(f"total_num_frame {total_num_frame}")  # 816000

    add_error_keypoints_percentage = (total_num_add_error_keypoints / total_num_keypoints) * 100
    error_hands_percentage = (total_num_add_error_hands / total_num_hands) * 100
    frame_error = (len(frame_error) / total_num_frame) * 100
    data_error = (len(row_error) / num_sample) * 100

    print(f"Simulated keypoints error percentage = {add_error_keypoints_percentage}%, "
          f"Simulated hand error percentage = {error_hands_percentage}%, rectified_data = {data_error}% and "
          f"Simulated frame error percentage = {frame_error}%,"
          f"where total_num_keypoints = {total_num_keypoints}, "
          f"Simulated keypoints error count = {total_num_add_error_keypoints}")
