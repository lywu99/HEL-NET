# CUDA_VISIBLE_DEVICES=0 python train_unet.py
import sys
import os
from optparse import OptionParser
import numpy as np

import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.optim import lr_scheduler

from utils import get_images, get_images_t

from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import copy
# from logger import Logger
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from log import *

from nets.vision_mamba import MambaUnet
from nets.config_mamba import get_config, add_mamba_args

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parser = OptionParser()
parser = OptionParser()
parser.add_option('-e', '--epochs', dest='epochs', default=600, type='int',
                  help='number of epochs')
parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                  type='int', help='batch size')
parser.add_option('-l', '--learning-rate', dest='lr', default=0.0002,  #0.0002
                  type='float', help='learning rate')
parser.add_option('-r', '--resume', dest='resume',
                  default=False, help='resume file model')
parser.add_option('-p', '--log-dir', dest='logdir', default='./logse/results/mambaunet/',
                  type='str', help='tensorboard log')
parser.add_option('-m', '--model-dir', dest='modeldir', default='./models',
                  type='str', help='models stored')
parser.add_option('-n', '--net-name', dest='netname', default='mambaunet',
                  type='str', help='net name, unet')
# parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
#                       default=False, help='preprocess input images')
# parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
#                   default='1', help='preprocess input images')
parser.add_option('-g', '--preprocess', dest='preprocess', action='store_true',
                  default='2', help='preprocess input images')
parser.add_option('-i', '--healthy-included', dest='healthyincluded', action='store_true',
                  default=False, help='include healthy images')
parser.add_option('-a', '--active-learning', dest='al', action='store_true',
                  default=False, help='whether to use active learning')

(args, _) = parser.parse_args()


# logger = Logger('./logs', args.logdir)
dir_checkpoint = args.modeldir
net_name = args.netname
lesion_dice_weights = [0., 0., 0., 0.]
lesions = ['ex', 'he', 'ma', 'se']

# image_dir_train = 'DR/data'
image_dir_train = './data'

softmax = nn.Softmax(1)


def eval_model(model, eval_loader, criterion_EX,criterion_SE,criterion_HE,criterion_MA):
    model.eval()

    eval_tot = len(eval_loader)
    eval_loss_ce = 0.
    ap = 0.
    aupr = 0.

    soft_masks_all = []
    true_masks_all = []


    with torch.set_grad_enabled(False):
        for inputs, true_masks in eval_loader:

            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)


            # if net_name == 'unet':
            #     masks_pred = model(inputs)

            masks_pred = model(inputs)

            masks_pred2 = masks_pred[:, 1, :, :]
            true_masks2 = true_masks[:, 1, :, :]
            masks_pred_flat2 = masks_pred2.reshape(-1)
            true_masks_flat2 = true_masks2.reshape(-1)
            loss_1 = criterion_EX(masks_pred_flat2, true_masks_flat2)

            # masks_pred3 = x3.permute(0, 2, 3, 1)
            masks_pred3 = masks_pred[:, 2, :, :]
            true_masks3 = true_masks[:, 2, :, :]
            masks_pred_flat3 = masks_pred3.reshape(-1)
            true_masks_flat3 = true_masks3.reshape(-1)
            loss_2 = criterion_SE(masks_pred_flat3, true_masks_flat3)

            # masks_pred4 = x4.permute(0, 2, 3, 1)
            masks_pred4 = masks_pred[:, 3, :, :]
            true_masks4 = true_masks[:, 3, :, :]
            masks_pred_flat4 = masks_pred4.reshape(-1)
            true_masks_flat4 = true_masks4.reshape(-1)
            loss_3 = criterion_HE(masks_pred_flat4, true_masks_flat4)

            # masks_pred5 = x5.permute(0, 2, 3, 1)
            masks_pred5 = masks_pred[:, 4, :, :]
            true_masks5 = true_masks[:, 4, :, :]
            masks_pred_flat5 = masks_pred5.reshape(-1)
            true_masks_flat5 = true_masks5.reshape(-1)
            loss_4 = criterion_MA(masks_pred_flat5, true_masks_flat5)


            eval_loss_ce += (loss_1+loss_2+loss_3+loss_4)/4


            masks_pred_softmax = torch.sigmoid(masks_pred)

            n_number = 4
            masks_soft = masks_pred_softmax[:, 1:, :, :].cpu().numpy()

            true_masks = torch.where(true_masks[:, 1:, :, :] > 0.5, 1,0)
            masks_true = true_masks.cpu().numpy()


            soft_masks_all.extend(masks_soft)
            true_masks_all.extend(masks_true)

        soft_masks_all = np.array(soft_masks_all)
        true_masks_all = np.array(true_masks_all)
        predicted = np.transpose(soft_masks_all, (1, 0, 2, 3))
        predicted = predicted.round(2)
        gt = np.transpose(true_masks_all, (1, 0, 2, 3))
        predicted = np.reshape(predicted, (predicted.shape[0], -1))
        gt = np.reshape(gt, (gt.shape[0], -1))

        for i in range(n_number):
            m = average_precision_score(gt[i], predicted[i])
            m.astype('float32')
            ap += m

            precision, recall, _ = precision_recall_curve(gt[i], predicted[i]) #wly add
            aupr += auc(recall, precision)


        return eval_loss_ce / eval_tot, ap / n_number, aupr/n_number


def denormalize(inputs):
    if net_name == 'unet':
        return (inputs * 255.).to(device=device, dtype=torch.uint8)
    else:
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).to(device)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).to(device)
        return ((inputs * std[None, :, None, None] + mean[None, :, None, None]) * 255.).to(device=device,
                                                                                           dtype=torch.uint8)

def train_model(model, train_loader, eval_loader, criterion_EX,criterion_SE,criterion_HE,criterion_MA, optimizer, scheduler, batch_size,logger,
                num_epochs=5, start_epoch=0, start_step=0):

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model,device_ids = [0, 1])
    model.to(device=device)
    tot_step_count = start_step

    output_dir = args.logdir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    best_ap = 0.
    best_dice = 0.
    best_aupr = 0.

    dir_checkpoint = output_dir + '/weights'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, start_epoch + num_epochs))
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, start_epoch + num_epochs))
        # scheduler.step()
        model.train()
        loss_sum = 0
        epoch_loss_ce = 0
        epoch_losses_dice = [0, 0, 0, 0]
        N_train = len(train_dataset)
        batch_step_count = 0
        vis_images = []
        for inputs, true_masks in tqdm(train_loader):

            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)

            # if net_name == 'unet':
            #     masks_pred = model(inputs)

            masks_pred = model(inputs)

            masks_pred2 = masks_pred[:, 1, :, :]
            true_masks2 = true_masks[:, 1, :, :]
            masks_pred_flat2 = masks_pred2.reshape(-1)
            true_masks_flat2 = true_masks2.reshape(-1)
            loss_1 = criterion_EX(masks_pred_flat2, true_masks_flat2)

            masks_pred3 = masks_pred[:, 2, :, :]
            true_masks3 = true_masks[:, 2, :, :]
            masks_pred_flat3 = masks_pred3.reshape(-1)
            true_masks_flat3 = true_masks3.reshape(-1)
            loss_2 = criterion_SE(masks_pred_flat3, true_masks_flat3)

            masks_pred4 = masks_pred[:, 3, :, :]
            true_masks4 = true_masks[:, 3, :, :]
            masks_pred_flat4 = masks_pred4.reshape(-1)
            true_masks_flat4 = true_masks4.reshape(-1)
            loss_3 = criterion_HE(masks_pred_flat4, true_masks_flat4)

            masks_pred5 = masks_pred[:, 4, :, :]
            true_masks5 = true_masks[:, 4, :, :]
            masks_pred_flat5 = masks_pred5.reshape(-1)
            true_masks_flat5 = true_masks5.reshape(-1)
            loss_4 = criterion_MA(masks_pred_flat5, true_masks_flat5)

            ce_weight = 1.
            loss = (loss_1+loss_2+loss_3+loss_4)/4


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss

            batch_step_count += 1
            tot_step_count += 1

        scheduler.step()
        num = len(train_loader)
        loss_ave = loss_sum / num
        print("loss_train:{}".format(loss_ave))
        logger.info("loss_train:{}".format(loss_ave))

        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)

        if (epoch + 1) % 1 == 0:
            loss_val, ap, aupr = eval_model(model, eval_loader, criterion_EX, criterion_SE, criterion_HE, criterion_MA)
            # loss_val,ap = func_eval_model(model, eval_loader, criterion_EX,criterion_SE,criterion_HE,criterion_MA)
            print("loss_val:{}  ".format(loss_val))
            # print("AP_val:{}".format(ap))
            print("AP_val:{} \t aupr_val:{}".format(ap, aupr))

            logger.info("loss_val:{}  ".format(loss_val))
            logger.info("AP_val:{} \t aupr_val:{}".format(ap, aupr))
            # with open(os.path.join(output_dir, "aupr_during_learning_" + args.model_name +  ".txt"), 'a') as f:


            if ap > best_ap:
                best_ap = ap
                state = {
                    'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    # 'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(state, \
                           os.path.join(dir_checkpoint, 'model_AP.pth.tar'))

                print("Epoch:{0} Best model_AP saved !!!".format(epoch))
                logger.info("Epoch:{0} Best model_AP saved !!!".format(epoch))

            if aupr > best_aupr:
                best_aupr = aupr
                state = {
                    'epoch': epoch,
                    'step': tot_step_count,
                    'state_dict': model.state_dict(),
                    # 'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                torch.save(state, \
                           os.path.join(dir_checkpoint, 'model_AUPR.pth.tar'))
                print("Epoch:{0} Best model_AUPR saved !!!".format(epoch))
                logger.info("Epoch:{0} Best model_AUPR saved !!!".format(epoch))


if __name__ == '__main__':

    output_dir = args.logdir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    logger = get_logger(name = args.netname, root= output_dir)
    logger.info(args)

    # if net_name == 'unet':
    #     model = UNet(n_channels=3, n_classes=5)
    if net_name == 'mambaunet':
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_size', type=int,
                            default=800, help='input patch size of network input')
        parser.add_argument("--num_classes", default=5, type=int)
        add_mamba_args(parser)
        args1 = parser.parse_args()
        config_mamba = get_config(args1)
        model = MambaUnet(config_mamba, img_size=args1.img_size,
                          num_classes=args1.num_classes)
        model.load_from(config_mamba, device)

    # model.to(device=device)

    # logger.info(model)
    n_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad  # p.numel() 表示返回张量中元素的总数量，即张量的大小。
    )
    logger.info("number of params:{}".format(n_parameters))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            start_step = checkpoint['step']
            model.load_state_dict(checkpoint['state_dict'])
            print('Model loaded from {}'.format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0
        start_step = 0


    train_image_paths, train_mask_paths = get_images(image_dir_train, args.preprocess, phase='train')
    eval_image_paths, eval_mask_paths = get_images(image_dir_train, args.preprocess, phase='eval')

    # train_image_paths, train_mask_paths = get_images_t(image_dir_train, args.preprocess, phase='train')
    # eval_image_paths, eval_mask_paths = get_images_t(image_dir_train, args.preprocess, phase='eval')

    train_dataset = IDRIDDataset(train_image_paths, train_mask_paths, 4, mode='train',augmentation_prob=0.6)
    eval_dataset = IDRIDDataset(eval_image_paths, eval_mask_paths, 4, mode='val',augmentation_prob=0)

    train_loader = DataLoader(train_dataset, args.batchsize, shuffle=True)
    eval_loader = DataLoader(eval_dataset, args.batchsize, shuffle=False)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    criterion_EX = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10.]).to(device))
    criterion_SE = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([40.]).to(device))
    criterion_HE = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10.]).to(device))
    criterion_MA = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([40.]).to(device))


    train_model(model, train_loader, eval_loader, criterion_EX,criterion_SE,criterion_HE,criterion_MA, optimizer, scheduler, args.batchsize,logger,
                num_epochs=args.epochs, start_epoch=start_epoch, start_step=start_step)
