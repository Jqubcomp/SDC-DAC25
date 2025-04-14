# This file contains code adapted from OpenFWI (https://github.com/lanl/OpenFWI)
# Licensed under the BSD 3-Clause License

import torch
from dataset import S_dataset
from vsnet import VSNet
from torch.utils.data import RandomSampler, DataLoader, random_split
import os
import utils
import transforms as T
import datetime

import numpy as np
import random
import time
import pytorch_ssim
import torchvision
from torchvision.transforms import Compose
import json
import sys
import torch.nn as nn
from scheduler import WarmupMultiStepLR
import wandb

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Encoder-Decoder Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='curvefault-a', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')
    parser.add_argument('--proj_name', default='UB_Diff_curvefault-a', type=str, help='wandb project name')
    
    parser.add_argument('-o', '--output-path',default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--num_data', default=48000, type=int, help='number of velocity maps')
    parser.add_argument('-n', '--save-name', default='training', help='folder name for this experiment')
    parser.add_argument('--latent_dim', default=128, type=int, help='latent dimension')
    # Path related    
    parser.add_argument('-l', '--log-path', default='./log', help='path to parent folder to save logs')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')
    parser.add_argument('-td', '--train-data', default='./seismic_data/', help='training seismic data path')
    parser.add_argument('-tl', '--train-label', default='./velocity_map/',help='training velocity map path')
    
    # Model related
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    # Training related
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.98, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('-eb', '--epoch_block', type=int, default=20, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=20, help='number of saved block')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--val_every', default=20, type=int)
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')

    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=1.0)
    parser.add_argument('-ls', '--lambda_s', type=float, default=1.0)
    parser.add_argument('-lv', '--lambda_v', type=float, default=1.0)

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')

    args.epochs = args.epoch_block * args.num_block
    print("Total epochs", args.epochs)

    if args.resume:
        args.resume = os.path.join(args.output_path, args.resume)

    return args


step = 0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, criterion, optimizer, lr_scheduler,
                    dataloader, device, epoch, print_freq, ctx, args):
    global step
    model.train()

    label_list, label_pred_list= [], [] # store denormalized predcition & gt in numpy
    label_tensor, label_pred_tensor = [], [] # store normalized prediction & gt in tensor

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)

    for data, label in metric_logger.log_every(dataloader, print_freq, header):
        start_time = time.time()
        optimizer.zero_grad()
        data, label = data.to(device, torch.float), label.to(device, torch.float)

        label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
        label_list.append(label_np)
        label_tensor.append(label)

        #print("label",label.shape)
        pred_seis, pred_vel = model(label)
        label_pred_tensor.append(pred_vel)

        #loss
        #seis loss
        seis_loss, seis_loss_g1v, seis_loss_g2v = criterion(pred_seis, data)

        #vel_loss
        vel_loss, vel_loss_g1v, vel_loss_g2v = criterion(pred_vel, label)


        loss = args.lambda_s * seis_loss + args.lambda_v * vel_loss

        loss.backward()
        optimizer.step()


        loss_val = loss.item()
        loss_s_val = seis_loss.item()
        loss_v_val = vel_loss.item()

        batch_size = data.shape[0]
        metric_logger.update(loss=loss_val, loss_s=loss_s_val,
            loss_v=loss_v_val, lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        if args.use_wandb:
            wandb.log({"loss": loss_val,"loss_s": loss_s_val, 'loss_v': loss_v_val})

        step += 1
        lr_scheduler.step()
        print(f"Epoch {epoch+1}, Learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"Train batch Loss at {epoch+1}: {loss_val}")
        print(f"Train batch Loss_s at {epoch+1}: {loss_s_val}")
        print(f"Train batch Loss_v at {epoch+1}: {loss_v_val}")

    # label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    print(f'Train SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}')  # (-1, 1) to (0, 1)


def evaluate(model, criterion, dataloader, device, ctx, args, epoch):
    model.eval()

    label_list, label_pred_list = [], []  # store denormalized predcition & gt in numpy
    label_tensor, label_pred_tensor = [], []  # store normalized prediction & gt in tensor

    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Test:'
    with torch.no_grad():
        for data, label in metric_logger.log_every(dataloader, 20, header):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)
            label_list.append(label_np)
            label_tensor.append(label)

            pred_seis, pred_vel = model(label)

            label_pred_tensor.append(pred_vel)

            # seis loss
            seis_loss, seis_loss_g1v, seis_loss_g2v = criterion(pred_seis, data)

            # vel_loss
            vel_loss, vel_loss_g1v, vel_loss_g2v = criterion(pred_vel, label)

            loss = args.lambda_s * seis_loss + args.lambda_v * vel_loss
            metric_logger.update(loss=loss.item(),
                                 loss_s=seis_loss.item(),
                                 loss_v=vel_loss.item(),
                                 )
            print(f"Val Loss at {epoch+1}: {loss.item()}")
            print(f"Val Loss_s at {epoch+1}: {seis_loss.item()}")
            print(f"Val Loss_v at {epoch+1}: {vel_loss.item()}")

    # Gather the statTotal time:s from all processes
    metric_logger.synchronize_between_processes()
    print(' * Loss {loss.global_avg:.8f}\n'.format(loss=metric_logger.loss))
    
    if args.use_wandb:
        wandb.log({"test loss": metric_logger.loss.global_avg,
               "test loss_s": metric_logger.loss_s.global_avg, 
               'test loss_v': metric_logger.loss_v.global_avg})

    label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    ssim = ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)
    print(f'Validation SSIM: {ssim}')  # (-1, 1) to (0, 1)

    return metric_logger.loss.global_avg, ssim

def main(args):
    global step

    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    utils.mkdir(args.output_path)  # create folder to store checkpoints
    if args.use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project= args.proj_name,
            name = args.save_name,
        )

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    with open('dataset_config.json') as f:
        try:
            ctx = json.load(f)[args.dataset]
        except KeyError:
            print('Unsupported dataset.')
            sys.exit()

    if args.file_size is not None:
        ctx['file_size'] = args.file_size

        # Create dataset and dataloader
    print('Loading data')
    print('Loading training data')

    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=args.k), T.log_transform(ctx['data_max'], k=args.k))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])

    #create dataset
    if args.dataset not in ['flatfault-a', 'curvefault-a', 'flatfault-b', 'curvefault-b']:
        fault_fam = False
    else:
        fault_fam = True

    train_dataset = S_dataset(args.train_data, args.train_label,transform_data, transform_label, pre_load=True, fault_fam=fault_fam)
    train_size = args.num_data
    
    test_size = len(train_dataset) - train_size
    dataset_train, dataset_test = random_split(train_dataset, [train_size, test_size])

    print("Training data:", len(dataset_train))
    assert len(dataset_train) == train_size

    print("Testing data:", len(dataset_test))

    print('Creating data loaders')
    train_sampler = RandomSampler(dataset_train)
    test_sampler = RandomSampler(dataset_test)
    
    #create dataloader
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=False)

    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True, drop_last=False)

    print('Creating model')
    model = VSNet(in_channels=1, out_channels_s=5, out_channels_v=1, dim5=args.latent_dim).to(device)

    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()

    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
    if len(args.lr_milestones) == 0:
        args.lr_milestones = [(epo + 1) for epo in range(args.epochs) if (epo + 1) % 10 == 0]

    lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
        warmup_iters=warmup_iters, warmup_factor=1e-5)
    
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        step = checkpoint['step']
        lr_scheduler.milestones = lr_milestones

    print('Start training')
    start_time = time.time()
    best_ssim = 0
    best_loss = 10
    chp = 1
    for epoch in range(args.start_epoch, args.epochs):
        print('=' * 50)
        print(f"Epoch: {epoch + 1}")
        train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader_train,
                        device, epoch, args.print_freq, ctx, args) #train vel with all data 
        if (epoch + 1) % args.val_every == 0:
            loss, ssim = evaluate(model, criterion, dataloader_test, device, ctx, args, epoch)

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'step': step,
                'args': args
                }
            if ssim > best_ssim:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_path, 'check_point.pth'))
                print('saving checkpoint at epoch: ', epoch)
                chp = epoch
                best_loss = loss
                best_ssim = ssim
            # Save checkpoint every epoch block
            print('current best epoch: ', chp)
            print('current best loss: ', best_loss)
            print('current best ssim: ', best_ssim)
            if args.output_path and (epoch + 1) % args.epoch_block == 0:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_path, 'check_point_{}.pth'.format(epoch + 1)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    setup_seed(0)
    args = parse_args()
    print(args.resume)
    main(args)










