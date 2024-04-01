import pandas as pd
from rectification_utils import compute_angle_between, get_angle_to_rotate, rotate_joint, convert_data_to_python_rep, compute_statistics
from rectification_utils import HAND_IDENTIFIERS, IGNORE_COL_LIST


def check_thumb_flexion_direction(hand_pose):
    if hand_pose[17][0] > hand_pose[1][0]:
        if hand_pose[0][1] < hand_pose[1][1]:
            flexion_direction = "clockwise"
        else:
            flexion_direction = "anticlockwise"
    else:
        if hand_pose[0][1] < hand_pose[1][1]:
            flexion_direction = "anticlockwise"
        else:
            flexion_direction = "clockwise"
    return flexion_direction


def rectify_thumb_flexion_and_extension(hand_pose, flexion_direction, flexion_extension_ranges,
                                        paired_joints_index_groups, associated_joints_index_groups,
                                        alpha):
    num_rectified_keypoints = 0
    for i, pair in enumerate(paired_joints_index_groups):
        joint_to_refer_index = pair[0]
        joint_to_check_index = pair[1]
        extra_joint_to_refer_index = pair[2]

        joint_to_check = hand_pose[joint_to_check_index]
        joint_to_refer = hand_pose[joint_to_refer_index]
        extra_joint_to_refer = hand_pose[extra_joint_to_refer_index]

        joint_to_check_prime = [joint_to_check[0] - extra_joint_to_refer[0],
                                joint_to_check[1] - extra_joint_to_refer[1]]
        joint_to_refer_prime = [joint_to_refer[0] - extra_joint_to_refer[0],
                                joint_to_refer[1] - extra_joint_to_refer[1]]
        joint_to_refer_prime = [joint_to_refer_prime[0] - joint_to_check_prime[0],
                                joint_to_refer_prime[1] - joint_to_check_prime[1]]

        try:
            angle = compute_angle_between(joint_to_check_prime, joint_to_refer_prime)
        except ZeroDivisionError:
            print("Missing value found")
            continue

        if flexion_direction == "clockwise":
            valid_range = flexion_extension_ranges[i]
        else:
            valid_range = -flexion_extension_ranges[i][1], -flexion_extension_ranges[i][0]

        angle_to_rotate = get_angle_to_rotate(angle, valid_range)

        related_points_index = associated_joints_index_groups[i]
        if angle_to_rotate != 0:
            num_rectified_keypoints += 1
            joint_to_refer_prime_new = rotate_joint(joint_to_refer_prime, -angle_to_rotate * alpha, is_clock_wise=True)
            new_angle = compute_angle_between(joint_to_check_prime, joint_to_refer_prime_new)
            print(f"new_angle = {new_angle}")

            joint_to_refer_new = [joint_to_refer_prime_new[0] + joint_to_check_prime[0] + extra_joint_to_refer[0],
                                  joint_to_refer_prime_new[1] + joint_to_check_prime[1] + extra_joint_to_refer[1]]

            movement = joint_to_refer_new[0] - joint_to_refer[0], joint_to_refer_new[1] - joint_to_refer[1]

            for joint_index in related_points_index:
                point = hand_pose[joint_index]
                new_point = point[0] + movement[0], point[1] + movement[1]
                hand_pose[joint_index] = new_point
            hand_pose[joint_to_refer_index] = joint_to_refer_new

    return hand_pose, num_rectified_keypoints


def rectify_other_flexion_and_extension(hand_pose, flexion_extension_ranges, paired_joints_index_groups,
                                        associated_joints_index_groups, alpha):
    num_rectified_keypoints = 0
    for i, pair in enumerate(paired_joints_index_groups):

        joint_to_refer_index = pair[0]
        joint_to_check_index = pair[1]
        extra_joint_to_refer_index = pair[2]

        joint_to_refer = hand_pose[joint_to_refer_index]
        joint_to_check = hand_pose[joint_to_check_index]
        extra_joint_to_refer = hand_pose[extra_joint_to_refer_index]

        joint_to_check_prime = [joint_to_check[0] - extra_joint_to_refer[0],
                                joint_to_check[1] - extra_joint_to_refer[1]]
        joint_to_refer_prime = [joint_to_refer[0] - extra_joint_to_refer[0],
                                joint_to_refer[1] - extra_joint_to_refer[1]]
        joint_to_refer_prime = [joint_to_refer_prime[0] - joint_to_check_prime[0],
                                joint_to_refer_prime[1] - joint_to_check_prime[1]]

        try:
            angle = compute_angle_between(joint_to_check_prime, joint_to_refer_prime)
        except ZeroDivisionError:
            print("Missing value found")
            continue

        valid_range = [-max(flexion_extension_ranges[i]), max(flexion_extension_ranges[i])]

        angle_to_rotate = get_angle_to_rotate(angle, valid_range)

        related_joint_index = associated_joints_index_groups[i]

        if angle_to_rotate != 0:
            num_rectified_keypoints += 1
            joint_to_refer_prime_new = rotate_joint(joint_to_refer_prime, -angle_to_rotate * alpha, is_clock_wise=True)
            new_angle = compute_angle_between(joint_to_check_prime, joint_to_refer_prime_new)
            print(f"new_angle = {new_angle}")

            joint_to_refer_new = [joint_to_refer_prime_new[0] + joint_to_check_prime[0] + extra_joint_to_refer[0],
                                  joint_to_refer_prime_new[1] + joint_to_check_prime[1] + extra_joint_to_refer[1]]

            movement = joint_to_refer_new[0] - joint_to_refer[0], joint_to_refer_new[1] - joint_to_refer[1]

            for joint_index in related_joint_index:
                point = hand_pose[joint_index]
                new_point = point[0] + movement[0], point[1] + movement[1]
                hand_pose[joint_index] = new_point
            hand_pose[joint_to_refer_index] = joint_to_refer_new
    return hand_pose, num_rectified_keypoints


def rectify_finger_flexion_and_extension(dataset_path, fe_active_motion_table_path, alpha=0.5):
    dataset_ori = pd.read_csv(dataset_path)
    dataset = convert_data_to_python_rep(dataset_ori, IGNORE_COL_LIST)

    fe_active_motion_table = convert_data_to_python_rep(pd.read_csv(fe_active_motion_table_path), ['fingers'])

    fe_thumb_active_motion_table = fe_active_motion_table[fe_active_motion_table['fingers'] == 'thumb']
    fe_thumb_abduction_adduction_ranges = fe_thumb_active_motion_table['ranges'].tolist()
    fe_thumb_paired_joints_index_groups = fe_thumb_active_motion_table[
        'paired_joint_index'].tolist()  # [joint_to_check, joint_to_refer, centre]
    fe_thumb_associated_joints_index_groups = fe_thumb_active_motion_table['associated_joint_index'].tolist()

    fe_other_active_motion_table = fe_active_motion_table[fe_active_motion_table['fingers'] != 'thumb']
    fe_other_flexion_extension_ranges = fe_other_active_motion_table['ranges'].tolist()
    fe_other_paired_joints_index_groups = fe_other_active_motion_table['paired_joint_index'].tolist()
    fe_other_associated_joints_index_groups = fe_other_active_motion_table['associated_joint_index'].tolist()

    total_num_rectified_keypoints = 0
    total_num_rectified_hands = 0
    rectified_frame = []
    rectified_row = []
    for which_hand in ["right", "left"]:
        full_hand_landmark_names = []
        for i in range(len(HAND_IDENTIFIERS)):
            tmp = ["_".join([HAND_IDENTIFIERS[i], which_hand, "X"]), "_".join([HAND_IDENTIFIERS[i], which_hand, "Y"])]
            full_hand_landmark_names.append(tmp)

        for row_index in range(len(dataset)):
            for frame_index in range(len(dataset[full_hand_landmark_names[0][0]][row_index])):
                hand_pose = []
                print(f"Processing {which_hand} hand at {row_index} row {frame_index} frame")
                for i in range(len(full_hand_landmark_names)):
                    xy_pairs = []
                    for j in range(len(full_hand_landmark_names[i])):
                        value = dataset[full_hand_landmark_names[i][j]][row_index][frame_index]
                        xy_pairs.append(value)
                    hand_pose.append(xy_pairs)

                # Check the flexion direction for thumb
                flexion_direction = check_thumb_flexion_direction(hand_pose)

                new_hand_pose, num_rectified_keypoints_thumb = rectify_thumb_flexion_and_extension(
                    hand_pose, flexion_direction, fe_thumb_abduction_adduction_ranges, fe_thumb_paired_joints_index_groups,
                    fe_thumb_associated_joints_index_groups, alpha)
                new_hand_pose, num_rectified_keypoints_others = rectify_other_flexion_and_extension(
                    new_hand_pose, fe_other_flexion_extension_ranges, fe_other_paired_joints_index_groups,
                    fe_other_associated_joints_index_groups, alpha)

                if num_rectified_keypoints_thumb > 0 or num_rectified_keypoints_others > 0:
                    total_num_rectified_keypoints += num_rectified_keypoints_thumb
                    total_num_rectified_keypoints += num_rectified_keypoints_others
                    total_num_rectified_hands += 1
                    if [row_index, frame_index] not in rectified_frame:
                        rectified_frame.append([row_index, frame_index])
                    if row_index not in rectified_row:
                        rectified_row.append(row_index)

                for i in range(len(full_hand_landmark_names)):
                    for j in range(len(full_hand_landmark_names[i])):
                        dataset[full_hand_landmark_names[i][j]][row_index][frame_index] = new_hand_pose[i][j]

    num_frame_per_sample = len(dataset.at[1, 'leftEar_X'])
    compute_statistics(num_frame_per_sample, len(dataset), total_num_rectified_keypoints, total_num_rectified_hands,
                       rectified_frame, rectified_row)

    return dataset
