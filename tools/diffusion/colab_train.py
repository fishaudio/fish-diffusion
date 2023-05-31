import argparse
import glob
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="path to config file")
parser.add_argument(
    "--pretrained", required=False, help="path to pretrained model checkpoint"
)
parser.add_argument("--resume", required=False, help="resume from checkpoint")
parser.add_argument("--resume-id", type=str, default=None, help="Wandb run id.")
parser.add_argument("--entity", type=str, default=None, help="Wandb entity.")
parser.add_argument("--name", type=str, default=None, help="Wandb run name.")
parser.add_argument(
    "--only-train-speaker-embeddings",
    action="store_true",
    required=False,
    help="Only train speaker embeddings.",
)
parser.add_argument("--dest-path", type=str, default=None, help="GDrive destination.")
parser.add_argument(
    "--tensorboard",
    action="store_true",
    required=False,
    help="enable tensorboard logging (recommended for colab), default is wandb",
)

args = parser.parse_args()

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def copy_files(src, dest):
    if not os.path.exists(src):
        logging.warning("%s does not exist, skipping %s", src, src)
        return
    else:
        for filename in os.listdir(src):
            if filename == "checkpoints":
                continue
            src_file = os.path.join(src, filename)
            dest_file = os.path.join(dest, os.path.relpath(src_file, src))
            if not os.path.exists(os.path.dirname(dest_file)):
                os.makedirs(os.path.dirname(dest_file))
                logging.info("Made %s!", os.path.dirname(dest_file))
            if os.path.isfile(src_file):
                src_stat = os.stat(src_file)
                if os.path.exists(dest_file):
                    dest_stat = os.stat(dest_file)
                    if src_stat.st_size == dest_stat.st_size and math.floor(
                        src_stat.st_mtime
                    ) <= math.floor(dest_stat.st_mtime):
                        continue
                    src_hash = hashlib.md5(open(src_file, "rb").read()).hexdigest()
                    dest_hash = hashlib.md5(open(dest_file, "rb").read()).hexdigest()
                    if src_hash == dest_hash:
                        continue
                    shutil.copy2(src_file, dest_file)
                    logging.info("Copied %s to %s!", src_file, dest_file)
                else:
                    shutil.copy2(src_file, dest_file)
                    logging.info("Copied %s to %s!", src_file, dest_file)


def get_step(filename):
    match = re.search(r"step=(\d+)", filename)
    return int(match.group(1)) if match else -1


# logs path
fishsvc_logs_path = "logs"
# must match [HiFiSVC or DiffSVC]
arch = "DiffSVC"
# path to destination specific to model name
fishsvc_dest_path = args.dest_path

fishsvc_chkpt_path = os.path.join(fishsvc_logs_path, arch)

# start the train.py process
train_args = [
    "python",
    "tools/diffusion/train.py",
    "--config",
    args.config,
]

if args.pretrained:
    train_args.extend(["--pretrained", args.pretrained])
if args.resume:
    train_args.extend(["--resume", args.resume])
if args.resume_id:
    train_args.extend(["--resume-id", args.resume_id])
if args.entity:
    train_args.extend(["--entity", args.entity])
if args.name:
    train_args.extend(["--name", args.name])
if args.tensorboard:
    train_args.append("--tensorboard")
if args.only_train_speaker_embeddings:
    train_args.append("--only-train-speaker-embeddings")

train_process = subprocess.Popen(train_args)

while True:
    try:
        if not os.path.exists(fishsvc_chkpt_path):
            logging.warning("%s doesn't exist yet!", fishsvc_chkpt_path)
            time.sleep(30)
            continue
        if args.dest_path:
            # Synchronize the logs folder with the destination folder
            if not args.tensorboard:
                for subdir, dirs, files in os.walk(
                    os.path.join(fishsvc_logs_path, "wandb")
                ):
                    copy_files(
                        os.path.join(subdir), os.path.join(fishsvc_dest_path, subdir)
                    )
            else:
                for version_dir in glob.glob(
                    os.path.join(fishsvc_chkpt_path, "version_*")
                ):
                    copy_files(
                        os.path.join(version_dir),
                        os.path.join(fishsvc_dest_path, version_dir),
                    )

            # Synchronize the checkpoints folder with the destination models folder

            if not args.tensorboard:
                for runid_dir in glob.glob(os.path.join(fishsvc_chkpt_path, "*")):
                    if "version_*" not in runid_dir:
                        copy_files(
                            os.path.join(runid_dir, "checkpoints"),
                            os.path.join(fishsvc_dest_path, "models"),
                        )
            else:
                for version_dir in glob.glob(
                    os.path.join(fishsvc_chkpt_path, "version_*")
                ):
                    copy_files(
                        os.path.join(version_dir, "checkpoints"),
                        os.path.join(fishsvc_dest_path, "models"),
                    )

        # Keep only the last four checkpoints in the source checkpoint directory
        for d in os.listdir(fishsvc_chkpt_path):
            checkpoints_path = os.path.join(fishsvc_chkpt_path, d, "checkpoints")
            if os.path.exists(checkpoints_path):
                checkpoints = sorted(
                    [
                        os.path.join(checkpoints_path, f)
                        for f in os.listdir(checkpoints_path)
                    ],
                    key=get_step,
                    reverse=True,
                )
                for checkpoint in checkpoints[4:]:
                    if ".ipynb_checkpoints" not in checkpoint:
                        os.remove(checkpoint)
            else:
                logging.warning(
                    "No checkpoints folder found in %s yet, skipping cleanup", d
                )

        if args.dest_path:
            # Keep only the last four checkpoints in the destination checkpoint directory
            models_path = os.path.join(fishsvc_dest_path, "models")
            if os.path.exists(models_path):
                checkpoints = sorted(
                    [os.path.join(models_path, f) for f in os.listdir(models_path)],
                    key=get_step,
                    reverse=True,
                )
                for checkpoint in checkpoints[4:]:
                    os.remove(checkpoint)
            else:
                logging.warning(
                    "No models folder found in %s yet, skipping cleanup",
                    fishsvc_dest_path,
                )

        time.sleep(
            30
        )  # wait for 30 seconds before running the checkpoint copier script again
    except KeyboardInterrupt:
        train_process.terminate()  # stop the train.py process if the user interrupts the script
        break
