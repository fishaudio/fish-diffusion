import torch
import argparse
import subprocess
import os

def pretrain_ckpt():
    checkpoint = torch.load(args.pretrain)
    checkpoint['global_step'] = 0
    checkpoint['epoch'] = 0
    checkpoint['lr_schedulers'] = [{'step_size': 20000, 'gamma': 0.5, 'base_lrs': [0.0002], 'last_epoch': 0, 'verbose': False, '_step_count': 0, '_get_lr_called_within_step': False, '_last_lr': [0.0002]}]
    checkpoint['optimizers'] = None
    checkpoint["MixedPrecisionPlugin"]["_growth_tracker"] = 0
    checkpoint['optimizer_states'] = ''
    checkpoint['loops'] = {'fit_loop': {'state_dict': {}, 'epoch_loop.state_dict': {'_batches_that_stepped': 0}, 'epoch_loop.batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'is_last_batch': True}, 'epoch_loop.scheduler_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.batch_loop.state_dict': {}, 'epoch_loop.batch_loop.optimizer_loop.state_dict': {}, 'epoch_loop.batch_loop.optimizer_loop.optim_progress': {'optimizer': {'step': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'zero_grad': {'total': {'ready': 0, 'completed': 0, 'started': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0}}}, 'optimizer_position': 0}, 'epoch_loop.batch_loop.manual_loop.state_dict': {}, 'epoch_loop.batch_loop.manual_loop.optim_step_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.val_loop.state_dict': {}, 'epoch_loop.val_loop.dataloader_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.val_loop.epoch_loop.state_dict': {}, 'epoch_loop.val_loop.epoch_loop.batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'is_last_batch': False}, 'epoch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}}}, 'validate_loop': {'state_dict': {}, 'dataloader_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.state_dict': {}, 'epoch_loop.batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'is_last_batch': False}}, 'test_loop': {'state_dict': {}, 'dataloader_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.state_dict': {}, 'epoch_loop.batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'is_last_batch': False}}, 'predict_loop': {'state_dict': {}, 'dataloader_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.state_dict': {}, 'epoch_loop.batch_progress': {'total': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}, 'current': {'ready': 0, 'completed': 0, 'started': 0, 'processed': 0}}}}
    torch.save(checkpoint, args.output)
    return checkpoint

##parser
parser = argparse.ArgumentParser(description='Prepare data to use alongside a pretrain')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
parser.add_argument('--pretrain', type=str, required=True, help='Path to the pretrained checkpoint file')
parser.add_argument('--output', type=str, required=True, help='Path to the output checkpoint file')
args = parser.parse_args()

#This is to run training using a pretrain model to get better results faster
#    --pretrain takes the path of the model you want to use as pretrain
#    --config takes the config you want to use to train
#    --output saves the pretrained model in the path you like
#    Recommended lr is 0.0001 and decay step should be every 10000 or 5000
######

pretrain_ckpt()
answer = input("Do you want to start training? y/n: ")

if answer.lower() == "y":
    print("Training will start, please wait.")
    subprocess.run(["python", "train.py", "--config", args.config, "--resume", args.output])
else:
    print("Exiting without starting training...")
    os.remove(args.output)
