import pandas as pd

from rectifications.flexion_extension_rectification import check_thumb_flexion_direction
from rectifications.rectification_utils import compute_angle_between, get_angle_to_rotate, rotate_joint, \
    compute_simulation_statistics, convert_data_to_python_rep

from rectification_utils import HAND_IDENTIFIERS, IGNORE_COL_LIST


def simulate_each_hand_abduction_and_addiction_error(hand_pose, which_hand, abduction_adduction_ranges,
                                                     paired_joints_index_groups, associated_joints_index_groups,
                                                     fake_error_angle):
    succ_add_incorrect_joints = 0

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
        act_error_angle = get_angle_to_rotate(angle, valid_range)

        # Check if the joint is correct
        if act_error_angle == 0:
            # Simulate error at the checked joint
            joint_to_check_prime_new = rotate_joint(joint_to_check_prime, fake_error_angle, is_clock_wise=True)
            new_angle = compute_angle_between(joint_to_check_prime_new, joint_to_refer_prime)
            new_joint_error = get_angle_to_rotate(new_angle, valid_range)
            if new_joint_error > 0:
                print(f"new joint error: {new_joint_error}")
                succ_add_incorrect_joints += 1

            joint_to_check_new = joint_to_check_prime_new[0] + centre[0], joint_to_check_prime_new[1] + centre[1]
            hand_pose[joint_index_to_check] = joint_to_check_new
            # Remain the related joints in the unattached state
            movement = joint_to_check_new[0] - joint_to_check[0], joint_to_check_new[1] - joint_to_check[1]
            for joint_index in related_joints_index:
                joint = hand_pose[joint_index]
                new_joint = joint[0] + movement[0], joint[1] + movement[1]
                hand_pose[joint_index] = new_joint

    return hand_pose, succ_add_incorrect_joints


def simulate_thumb_flexion_and_extension_error(hand_pose, flexion_direction, flexion_extension_ranges,
                                               paired_joints_index_groups, associated_joints_index_groups,
                                               fake_error_angle):
    succ_add_incorrect_joints = 0
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

        act_error_angle = get_angle_to_rotate(angle, valid_range)

        related_points_index = associated_joints_index_groups[i]
        if act_error_angle == 0:
            joint_to_refer_prime_new = rotate_joint(joint_to_refer_prime, fake_error_angle, is_clock_wise=True)
            new_angle = compute_angle_between(joint_to_check_prime, joint_to_refer_prime_new)
            new_joint_error = get_angle_to_rotate(new_angle, valid_range)
            if new_joint_error > 0:
                print(f"new joint error: {new_joint_error}")
                succ_add_incorrect_joints += 1

            joint_to_refer_new = [joint_to_refer_prime_new[0] + joint_to_check_prime[0] + extra_joint_to_refer[0],
                                  joint_to_refer_prime_new[1] + joint_to_check_prime[1] + extra_joint_to_refer[1]]

            movement = joint_to_refer_new[0] - joint_to_refer[0], joint_to_refer_new[1] - joint_to_refer[1]

            for joint_index in related_points_index:
                point = hand_pose[joint_index]
                new_point = point[0] + movement[0], point[1] + movement[1]
                hand_pose[joint_index] = new_point
            hand_pose[joint_to_refer_index] = joint_to_refer_new

    return hand_pose, succ_add_incorrect_joints


def simulate_other_flexion_and_extension_error(hand_pose, flexion_extension_ranges, paired_joints_index_groups,
                                               associated_joints_index_groups, fake_error_angle):
    succ_add_incorrect_joints = 0
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

        act_error_angle = get_angle_to_rotate(angle, valid_range)

        related_joint_index = associated_joints_index_groups[i]

        if act_error_angle == 0:
            joint_to_refer_prime_new = rotate_joint(joint_to_refer_prime, fake_error_angle, is_clock_wise=True)
            new_angle = compute_angle_between(joint_to_check_prime, joint_to_refer_prime_new)
            new_joint_error = get_angle_to_rotate(new_angle, valid_range)
            if new_joint_error > 0:
                print(f"new joint error: {new_joint_error}")
                succ_add_incorrect_joints += 1

            joint_to_refer_new = [joint_to_refer_prime_new[0] + joint_to_check_prime[0] + extra_joint_to_refer[0],
                                  joint_to_refer_prime_new[1] + joint_to_check_prime[1] + extra_joint_to_refer[1]]

            movement = joint_to_refer_new[0] - joint_to_refer[0], joint_to_refer_new[1] - joint_to_refer[1]

            for joint_index in related_joint_index:
                point = hand_pose[joint_index]
                new_point = point[0] + movement[0], point[1] + movement[1]
                hand_pose[joint_index] = new_point
            hand_pose[joint_to_refer_index] = joint_to_refer_new

    return hand_pose, succ_add_incorrect_joints


def simulate_joint_error(dataset_path, aa_active_motion_table_path, fe_active_motion_table_path, add_error_per=0.5,
                         fake_error_angle=90, add_num_error_hand=None):
    dataset_ori = pd.read_csv(dataset_path)
    dataset = convert_data_to_python_rep(dataset_ori, IGNORE_COL_LIST)

    aa_active_motion_table = convert_data_to_python_rep(pd.read_csv(aa_active_motion_table_path))
    abduction_adduction_ranges = aa_active_motion_table['ranges'].tolist()
    aa_paired_joints_index_groups = aa_active_motion_table[
        'paired_joint_index'].tolist()  # [joint_to_check, joint_to_refer, centre]
    aa_associated_joints_index_groups = aa_active_motion_table['associated_joint_index'].tolist()

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

    total_num_add_keypoints_error = 0
    total_num_add_hands_error = 0
    frame_error = []
    row_error = []
    num_frame_per_sample = len(dataset.at[1, 'leftEar_X'])
    num_sample = len(dataset)
    total_num_hands = 2 * num_frame_per_sample * num_sample

    for which_hand in ["right", "left"]:
        full_hand_landmark_names = []
        for i in range(len(HAND_IDENTIFIERS)):
            tmp = ["_".join([HAND_IDENTIFIERS[i], which_hand, "X"]), "_".join([HAND_IDENTIFIERS[i], which_hand, "Y"])]
            full_hand_landmark_names.append(tmp)

        for row_index in range(len(dataset)):
            for frame_index in range(len(dataset[full_hand_landmark_names[0][0]][row_index])):
                hand_pose = []
                if add_num_error_hand is not None:
                    if total_num_add_hands_error == add_num_error_hand:
                        compute_simulation_statistics(num_frame_per_sample, num_sample, total_num_add_keypoints_error,
                                                      total_num_add_hands_error,
                                                      frame_error, row_error)
                        return dataset
                elif int(total_num_add_hands_error / total_num_hands) == add_error_per:
                    compute_simulation_statistics(num_frame_per_sample, num_sample, total_num_add_keypoints_error,
                                                  total_num_add_hands_error,
                                                  frame_error, row_error)
                    return dataset

                print(f"Processing {which_hand} hand at {row_index} row {frame_index} frame")
                for i in range(len(full_hand_landmark_names)):
                    xy_pairs = []
                    for j in range(len(full_hand_landmark_names[i])):
                        value = dataset[full_hand_landmark_names[i][j]][row_index][frame_index]
                        xy_pairs.append(value)
                    hand_pose.append(xy_pairs)

                new_hand_pose, num_aa_error_keypoints = simulate_each_hand_abduction_and_addiction_error(
                    hand_pose, which_hand, abduction_adduction_ranges, aa_paired_joints_index_groups,
                    aa_associated_joints_index_groups, fake_error_angle=fake_error_angle)

                # Check the flexion direction for thumb
                flexion_direction = check_thumb_flexion_direction(new_hand_pose)

                new_hand_pose, num_fe_error_keypoints_thumb = simulate_thumb_flexion_and_extension_error(
                    new_hand_pose, flexion_direction, fe_thumb_abduction_adduction_ranges,
                    fe_thumb_paired_joints_index_groups,
                    fe_thumb_associated_joints_index_groups, fake_error_angle=fake_error_angle)
                new_hand_pose, num_fe_error_keypoints_others = simulate_other_flexion_and_extension_error(
                    new_hand_pose, fe_other_flexion_extension_ranges, fe_other_paired_joints_index_groups,
                    fe_other_associated_joints_index_groups, fake_error_angle=fake_error_angle)

                if num_aa_error_keypoints > 0 or num_fe_error_keypoints_thumb > 0 \
                        or num_fe_error_keypoints_others > 0:
                    total_num_add_keypoints_error += num_aa_error_keypoints
                    total_num_add_keypoints_error += num_fe_error_keypoints_thumb
                    total_num_add_keypoints_error += num_fe_error_keypoints_others

                    total_num_add_hands_error += 1
                    if [row_index, frame_index] not in frame_error:
                        frame_error.append([row_index, frame_index])
                    if row_index not in row_error:
                        row_error.append(row_index)

                for i in range(len(full_hand_landmark_names)):
                    for j in range(len(full_hand_landmark_names[i])):
                        dataset[full_hand_landmark_names[i][j]][row_index][frame_index] = new_hand_pose[i][j]

    compute_simulation_statistics(num_frame_per_sample, num_sample, total_num_add_keypoints_error,
                                  total_num_add_hands_error,
                                  frame_error, row_error)
    # rectified_keypoints_percentage = 2.8152732389252%, rectified_hands_percentage = 16.184191176470588%,
    # rectified_data = 97.52499999999999% and rectified_frame = 21.89142156862745%,
    # where total_num_keypoints = 44064000, num_rectified_keypoints = 1240522
    return dataset


if __name__ == "__main__":
    fe_motion_table_path = r"D:\Skeleton_based_SLR\active_motion\flexion_extension_ranges.csv"
    aa_motion_table_path = r"D:\Skeleton_based_SLR\active_motion\abduction_adduction_ranges.csv"
    # input_path = r"D:\Skeleton_based_SLR\datasets\oversample\balanced_WLASL100_SMOTE.csv"
    input_path = r"D:\Skeleton_based_SLR\datasets\rectified\combination\rectified_aafe_1_balanced_WLASL100_SMOTE.csv"

    error_dataset = simulate_joint_error(input_path, aa_motion_table_path, fe_motion_table_path,
                                         add_num_error_hand=81600)
    output_path = r"D:\Skeleton_based_SLR\datasets\JES\JES_1_balanced_WLASL100_SMOTE.csv"
    error_dataset.to_csv(output_path, index=False)
    print("saved at: " + output_path)

    error_dataset = simulate_joint_error(input_path, aa_motion_table_path, fe_motion_table_path,
                                         add_num_error_hand=163200)
    output_path = r"D:\Skeleton_based_SLR\datasets\JES\JES_2_balanced_WLASL100_SMOTE.csv"
    error_dataset.to_csv(output_path, index=False)
    print("saved at: " + output_path)

    error_dataset = simulate_joint_error(input_path, aa_motion_table_path, fe_motion_table_path,
                                         add_num_error_hand=244800)
    output_path = r"D:\Skeleton_based_SLR\datasets\JES\JES_3_balanced_WLASL100_SMOTE.csv"
    error_dataset.to_csv(output_path, index=False)
    print("saved at: " + output_path)

    error_dataset = simulate_joint_error(input_path, aa_motion_table_path, fe_motion_table_path,
                                         add_num_error_hand=326400)
    output_path = r"D:\Skeleton_based_SLR\datasets\JES\JES_4_balanced_WLASL100_SMOTE.csv"
    error_dataset.to_csv(output_path, index=False)
    print("saved at: " + output_path)

    error_dataset = simulate_joint_error(input_path, aa_motion_table_path, fe_motion_table_path,
                                         add_num_error_hand=408000)
    output_path = r"D:\Skeleton_based_SLR\datasets\JES\JES_4_balanced_WLASL100_SMOTE.csv"
    error_dataset.to_csv(output_path, index=False)
    print("saved at: " + output_path)
