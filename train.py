import os
import argparse
import random
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

from utils import __balance_val_split, __split_of_train_sequence, __log_class_statistics
from datasets.czech_slr_dataset import CzechSLRDataset
from siformer.model import SiFormer, SpoTer
from siformer.utils import train_epoch, evaluate, evaluate_top_k
from siformer.gaussian_noise import GaussianNoise

import time
import datetime
from statistics import mean


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="WLASL_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes to be recognized by the model")
    parser.add_argument("--batch_size", type=int, default=24, help="Number of batch size")
    parser.add_argument("--num_worker", type=int, default=24, help="Number of workers")
    parser.add_argument("--num_seq_elements", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="none",
                        help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=int, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    # Training time
    parser.add_argument("--record_training_time", type=bool, default=False,
                        help="Determines whether continuous statistics of training time should be record")

    # Model settings
    parser.add_argument("--attn_type", type=str, default='prob', help="The attention mechanism used by the model")
    parser.add_argument("--num_enc_layers", type=int, default=3, help="Determines the number of encoder layers")
    parser.add_argument("--num_dec_layers", type=int, default=2, help="Determines the number of decoder layers")
    parser.add_argument("--FIM", type=bool, default=True, help="Determines whether feature-isolated mechanism will be applied")
    parser.add_argument("--IA_encoder", type=bool, default=True, help="Determines whether input adaptive encoder will be used")
    parser.add_argument("--IA_decoder", type=bool, default=False, help="Determines whether input adaptive decoder will be used")
    parser.add_argument("--patience", type=int, default=1, help="Determines the patience for earlier exist")

    return parser


def train(args):
    # MARK: TRAINING PREPARATION AND MODULES

    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Cuda is available: True")
        device = torch.device("cuda")

    # Construct the model
    if args.FIM:
        slr_model = SiFormer(num_classes=args.num_classes, num_hid=args.num_seq_elements, attn_type=args.attn_type,
                              num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers, device=device,
                              IA_encoder=args.IA_encoder, IA_decoder=args.IA_decoder,
                              patience=args.patience)
    else:
        slr_model = SpoTer(num_classes=args.num_classes, num_hid=args.num_seq_elements,
                           num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers)
    slr_model.train(True)
    slr_model.to(device)

    # Construct the other modules
    cel_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(slr_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)  # 40, 60, 80

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    # MARK: DATA

    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = CzechSLRDataset(args.training_set_path, transform=transform, augmentations=True)

    # Validation set
    if args.validation_set == "from-file":
        val_set = CzechSLRDataset(args.validation_set_path)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, generator=g,
                                num_workers=args.num_worker)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, generator=g,
                                num_workers=args.num_worker)

    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        eval_set = CzechSLRDataset(args.testing_set_path)
        eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, generator=g,
                                 num_workers=args.num_worker)

    else:
        eval_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=args.num_worker)

    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print(
            "Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...")
        logging.info(
            "Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...")
    else:
        print("Starting " + args.experiment_name + "...")
        logging.info("Starting " + args.experiment_name + "...")

    print("Training using " + args.training_set_path + "...")
    logging.info("Training using " + args.training_set_path + "...")

    if args.validation_set == "from-file":
        print("Validation using " + args.validation_set_path + "...\n\n")
        logging.info("Validation using " + args.validation_set_path + "...\n\n")

    total_train_time = 0
    avg_train_time_sec_list = []
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, _, _, train_acc, avg_train_time = train_epoch(slr_model, train_loader, cel_criterion, optimizer,
                                                                  device, scheduler=scheduler)
        end_time = time.time()
        train_time = end_time - start_time

        if args.record_training_time:
            avg_train_time_sec_list.append(avg_train_time)
            total_train_time += train_time

        if val_loader:
            slr_model.train(False)
            _, _, val_acc = evaluate(slr_model, val_loader, device)
            slr_model.train(True)
            val_accs.append(val_acc)

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                torch.save(slr_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(
                    checkpoint_index) + ".pth")

            if val_acc > top_val_acc:
                top_val_acc = val_acc
                torch.save(slr_model, "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(
                    checkpoint_index) + ".pth")

                print(f'Save checkpoint for [{str(epoch + 1)}] as ' + "out-checkpoints/" + args.experiment_name
                      + "/checkpoint_v_" + str(checkpoint_index) + ".pth")
                logging.info(f'Save checkpoint for [{str(epoch + 1)}] as ' + "out-checkpoints/" + args.experiment_name
                             + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print(
                "[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(
                    train_acc))
            print(
                f"[{str(epoch + 1)}] AVG TRAIN time per sample (sec): {str(avg_train_time)} "
            )
            logging.info(
                "[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item() / len(train_loader)) + " acc: " + str(
                    train_acc))
            logging.info(
                f"[{str(epoch + 1)}] AVG TRAIN time per sample (sec): {str(avg_train_time)} "
            )

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  acc: " + str(val_acc))

                print("[" + str(epoch + 1) + "] VALIDATION  Top 5 acc: " + str(top_val_acc))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  Top 5 acc: " + str(top_val_acc))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc = 0, 0
            checkpoint_index += 1

        lr_progress.append(optimizer.param_groups[0]["lr"])

    if args.record_training_time:
        print(f"Total training time taken over {args.epochs} epochs: {str(datetime.timedelta(seconds=total_train_time))}")
        print(f"Average training time per sample: {str(mean(avg_train_time_sec_list[1:]))}")

        logging.info(f"Total training time taken over {args.epochs} epochs: {str(datetime.timedelta(seconds=total_train_time))}")
        logging.info(f"Average training time per sample: {str(mean(avg_train_time_sec_list[1:]))}")

    # MARK: TESTING
    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if eval_loader:
        for i in range(checkpoint_index):
            for checkpoint_id in ["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load(
                    "out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                tested_model.train(False)
                _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)
                _, _, top_val_acc = evaluate_top_k(slr_model, val_loader, device)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        print("\nThe top result was recorded at " + str(
            top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(
            top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")

    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True,
                   fontsize="xx-small")
        ax.grid()

        fig.savefig("out-img/" + args.experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_lr.png")

    # PLOT 2: Training time
    if args.record_training_time:
        fig1, ax2 = plt.subplots()
        ax2.plot(range(1, len(avg_train_time_sec_list) + 1), avg_train_time_sec_list,
                 label="AVG Training Time per sample")
        ax2.set(xlabel="Epoch", ylabel="Second", title="")
        ax2.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_tt.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    train(args)
