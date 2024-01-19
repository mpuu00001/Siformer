import pandas as pd
from rectification_utils import compute_angle_between, get_angle_to_rotate, rotate_joint, convert_data_to_python_rep, compute_statistics
from rectification_utils import HAND_IDENTIFIERS, IGNORE_COL_LIST


def rectify_each_hand_abduction_and_addiction(hand_pose, which_hand, abduction_adduction_ranges,
                                              paired_joints_index_groups, associated_joints_index_groups,
                                              alpha):
    num_rectified_keypoints = 0
    for i, pair in enumerate(paired_joints_index_groups):
        valid_range_ori = abduction_adduction_ranges[i]
        related_joints_index = associated_joints_index_groups[i]

        # Get the index for the joint to check and its references
        joint_index_to_check = pair[0]
        joint_index_to_refer = pair[1]
        centre_index = pair[2]

        # Get the joint to check and its references
        joint_to_check = hand_pose[joint_index_to_check]
        joint_to_refer = hand_pose[joint_index_to_refer]
        centre = hand_pose[centre_index]

        # Move the centre to be at the origin and move others
        joint_to_check_prime = joint_to_check[0] - centre[0], joint_to_check[1] - centre[1]
        joint_to_refer_prime = joint_to_refer[0] - centre[0], joint_to_refer[1] - centre[1]

        # Get the angle between
        try:
            angle = compute_angle_between(joint_to_check_prime, joint_to_refer_prime)
        except ZeroDivisionError:
            print("Missing value found")
            continue

        # Get the valid range define the joint angle
        if which_hand == "right":
            valid_range = -valid_range_ori[1], -valid_range_ori[0]
        else:
            valid_range = valid_range_ori

        # Compute the angle to rotate
        angle_to_rotate = get_angle_to_rotate(angle, valid_range)

        # Check if there are errors
        if angle_to_rotate != 0:
            num_rectified_keypoints += 1
            # Rectify the checked joint
            act_angle_to_rotate = angle_to_rotate * alpha
            joint_to_check_prime_new = rotate_joint(joint_to_check_prime, act_angle_to_rotate, is_clock_wise=True)
            new_angle = compute_angle_between(joint_to_check_prime_new, joint_to_refer_prime)
            print(f"new_angle = {new_angle}")

            joint_to_check_new = joint_to_check_prime_new[0] + centre[0], joint_to_check_prime_new[1] + centre[1]
            hand_pose[joint_index_to_check] = joint_to_check_new
            # Rectify the related joints
            movement = joint_to_check_new[0] - joint_to_check[0], joint_to_check_new[1] - joint_to_check[1]
            for joint_index in related_joints_index:
                joint = hand_pose[joint_index]
                new_joint = joint[0] + movement[0], joint[1] + movement[1]
                hand_pose[joint_index] = new_joint
    return hand_pose, num_rectified_keypoints


def rectify_finger_abduction_and_addiction(dataset_path, aa_active_motion_table_path, alpha=0.5):
    dataset_ori = pd.read_csv(dataset_path)
    dataset = convert_data_to_python_rep(dataset_ori, IGNORE_COL_LIST)

    aa_active_motion_table = convert_data_to_python_rep(pd.read_csv(aa_active_motion_table_path))
    abduction_adduction_ranges = aa_active_motion_table['ranges'].tolist()
    aa_paired_joints_index_groups = aa_active_motion_table['paired_joint_index'].tolist() # [joint_to_check, joint_to_refer, centre]
    aa_associated_joints_index_groups = aa_active_motion_table['associated_joint_index'].tolist()

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

                new_hand_pose, num_aa_rectified_keypoints = rectify_each_hand_abduction_and_addiction(
                    hand_pose, which_hand, abduction_adduction_ranges, aa_paired_joints_index_groups,
                    aa_associated_joints_index_groups, alpha)
                if num_aa_rectified_keypoints > 0:
                    total_num_rectified_keypoints += num_aa_rectified_keypoints
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
    # rectified_keypoints_percentage = 1.5515386710239651%, rectified_hands_percentage = 15.768872549019608%,
    # rectified_data = 97.375% and rectified_frame = 21.43235294117647%, where total_num_keypoints = 44064000,
    # num_rectified_keypoints = 683670
    return dataset


if __name__ == "__main__":
    motion_table_path = r"D:\Skeleton_based_SLR\active_motion\abduction_adduction_ranges.csv"
    input_path = r"D:\Skeleton_based_SLR\datasets\rectified\flexion_and_extension\rectified_fe_04_balanced_WLASL100_SMOTE.csv"  # rectified_data_percentage = 31.53% # rectified_keypoint_percentatge = 8%
    rectified_dataset = rectify_finger_abduction_and_addiction(input_path, motion_table_path, 0.4)
    output_path = r"D:\Skeleton_based_SLR\datasets\rectified\combination\rectified_aafe_04_balanced_WLASL100_SMOTE.csv"
    rectified_dataset.to_csv(output_path, index=False)

    # # alpha = 0.2
    # rectified_dataset = rectify_finger_abduction_and_addiction(input_path, motion_table_path, 0.2)
    # output_path_2 = r"D:\Skeleton_based_SLR\datasets\rectified\rectified_aa_02_balanced_WLASL100_SMOTE.csv"
    # rectified_dataset.to_csv(output_path_2, index=False)
    #
    # # alpha = 0.4
    # rectified_dataset = rectify_finger_abduction_and_addiction(input_path, motion_table_path, 0.4)
    # output_path_4 = r"D:\Skeleton_based_SLR\datasets\rectified\rectified_aa_04_balanced_WLASL100_SMOTE.csv"
    # rectified_dataset.to_csv(output_path_4, index=False)
    #
    # # alpha = 0.6
    # rectified_dataset = rectify_finger_abduction_and_addiction(input_path, motion_table_path, 0.6)
    # output_path_6 = r"D:\Skeleton_based_SLR\datasets\rectified\rectified_aa_06_balanced_WLASL100_SMOTE.csv"
    # rectified_dataset.to_csv(output_path_6, index=False)
    #
    # # alpha = 0.8
    # rectified_dataset = rectify_finger_abduction_and_addiction(input_path, motion_table_path, 0.8)
    # output_path_8 = r"D:\Skeleton_based_SLR\datasets\rectified\rectified_aa_08_balanced_WLASL100_SMOTE.csv"
    # rectified_dataset.to_csv(output_path_8, index=False)
    #
    # # alpha = 1
    # rectified_dataset = rectify_finger_abduction_and_addiction(input_path, motion_table_path, 1)
    # output_path_1 = r"D:\Skeleton_based_SLR\datasets\rectified\rectified_aa_1_balanced_WLASL100_SMOTE.csv"
    # rectified_dataset.to_csv(output_path_1, index=False)
