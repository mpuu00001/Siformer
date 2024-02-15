#!/bin/bash
#
#SBATCH --job-name=WLASL_spoter_R_9_b_PBEE
#SBATCH --time=24:00:00
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --partition=gpu.large

module add cuda/cudnn/8.9.6 python/3.10.5

python -m train --experiment_name WLASL_spoter_R_9_b_PBEE_6ecr_6dcr --training_set_path /ibm/gpfs/home/mupu0001/Skeleton_based_SLR/datasets/rectified/flexion_and_extension/rectified_fe_04_balanced_WLASL100_SMOTE.csv --experimental_train_split 0.8 --validation_set split-from-train --validation_set_size 0.2 --num_classes 100
