# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
import pdb
from pprint import pprint
from itertools import cycle

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models

from data.dataload import ENA_Dataset, SYN_Dataset, ENA_Dataset_unlabeled, ConcatDataset
from data.Transforms import get_transforms
import data.config as cfg
from sklearn.model_selection import train_test_split


# from data_utils.Desed import DESED
# from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from TestModel import _load_crnn
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics, get_f_measure_by_class
from models.CRNN_GRL import CRNN_fpn, CRNN, Predictor, Frame_Discriminator, Clip_Discriminator
# from DA.cdan import ConditionalDomainAdversarialLoss
from DA.dan import ConditionalDomainAdversarialLoss
# from DA.cdan import ConditionalDomainAdversarialLoss

from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from torch.autograd import Variable
import random
import torch.nn.functional as F
import collections

class Net_resnet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__() # necessary
        self.resnet = models.resnet18(pretrained=pretrained)
        # self.resnet = Resnet.resnet18(pretrained=pretrained)
        # convert last layer of resnet to 10 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, len(cfg.bird_list))
        # convert convolution to single channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.resnet(x)
        out = self.sigmoid(x)
        return out

class Net_vgg(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__() # necessary
        self.vgg = models.vgg19_bn(pretrained=pretrained)
        self.fc = nn.Linear(1000, len(cfg.classes))
        # convert convolution to single channel
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 1-to-3 channels
        x = torch.cat((x,x,x), 1)
        x = self.vgg(x)
        x = self.fc(x)
        out = self.sigmoid(x)
        return out
    
def print_ct_matrix(ct_matrix):
    print('\n'.join([''.join(['{:5}'.format(int(item)) for item in row]) for row in ct_matrix]))

def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1, optimizer_d=None, optimizer_crnn=None, c_epoch=None, rampup_value_adv=None):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:
    """
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    # We commented parts on betas and weight decay to match 2nd system of last year from Orange
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    # lr_adv = rampup_value_adv * rampdown_value * cfg.max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    # if c_epoch % 25 == 0:
    # lr = 0.001 * pow(0.5, c_epoch//25)
    # lr_adv = lr_adv * 0.1
    if c_epoch > 100:
        lr = lr * (0.5 ** (1 + ((c_epoch - 100) // 20)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['betas'] = (beta1, beta2)
        # param_group['weight_decay'] = weight_decay

    if optimizer_d != None:
        for param_group in optimizer_d.param_groups:
            # param_group['lr'] = lr * 0.1
            param_group['lr'] = lr * 0.1

    if optimizer_crnn != None:
        for param_group in optimizer_crnn.param_groups:
            param_group['lr'] = lr * 0.1
            # param_group['lr'] = lr


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    # for ema_params, params in zip(ema_model.parameters(), model.parameters()):
    #     ema_params.data.mul_(alpha).add_(1 - alpha, params.data)
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_model_state_dict = ema_model.state_dict()
        for entry in ema_model_state_dict.keys():
            ema_param = ema_model_state_dict[entry].clone().detach()
            param = model_state_dict[entry].clone().detach()
            new_param = (ema_param * alpha) + (param * (1. - alpha))
            ema_model_state_dict[entry] = new_param
        ema_model.load_state_dict(ema_model_state_dict)


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and (p.grad != None):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("gradient_flow.png")

### ICT
def get_current_consistency_weight(final_consistency_weight, epoch, step_in_epoch, total_steps_in_epoch, consistency_rampup_starts, consistency_rampup_ends):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - consistency_rampup_starts
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight * ramps.sigmoid_rampup(epoch, consistency_rampup_ends - consistency_rampup_starts )

def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, z, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y, z = x.data.cpu().numpy(), y.data.cpu().numpy(), z.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index,:])
    mixed_z = torch.Tensor(lam * z + (1 - lam) * z[index,:])

    mixed_x = Variable(mixed_x.cuda())
    mixed_y = Variable(mixed_y.cuda())
    mixed_z = Variable(mixed_z.cuda())
    return mixed_x, mixed_y, mixed_z, lam



def train_mt(train_unlabeled_loader, train_weak_loader, syn_loader, model, optimizer, c_epoch, ema_model=None, ema_predictor=None, mask_weak=None, mask_strong=None, adjust_lr=False, discriminator=None, optimizer_d=None, predictor=None, optimizer_crnn=None, ISP=False):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    domain_acc = []
    # ema_model = None
    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_unlabeled_loader)))
    start = time.time()
    # for i, (xb1, xb2) in enumerate(zip(train_loader, cycle(syn_loader))):
    # for epoch in range(num_epochs):
    # dataloader_iterator = iter(syn_loader)
    unlabeled_dataloader_iterator = iter(train_unlabeled_loader)
    weak_dataloader_iterator = iter(train_weak_loader)
    
    for i, data_syn in enumerate(syn_loader):

        try:
            data_unlabeled = next(unlabeled_dataloader_iterator)
        except StopIteration:
            unlabeled_dataloader_iterator = iter(train_unlabeled_loader)
            data_unlabeled = next(unlabeled_dataloader_iterator)
        
        try:
            data_weak = next(weak_dataloader_iterator)
        except StopIteration:
            weak_dataloader_iterator = iter(train_weak_loader)
            data_weak = next(weak_dataloader_iterator)

        ((unlabeled_batch_input, unlabeled_ema_batch_input), target_pl), filename = data_unlabeled

        ((weak_batch_input, weak_ema_batch_input), target), filename = data_weak
        target_weak = target.max(-2)[0]

        ((syn_batch_input, syn_ema_batch_input), syn_target), syn_filename = data_syn
        

        if (unlabeled_batch_input.shape[0] != syn_batch_input.shape[0]//2):
            continue
        if (weak_batch_input.shape[0] != syn_batch_input.shape[0]//2):
            continue

        batch_input = torch.cat((weak_batch_input, unlabeled_batch_input), dim=0)
        ema_batch_input = torch.cat((weak_ema_batch_input, unlabeled_ema_batch_input), dim=0)
        target_weak = torch.cat((target_weak, target_pl), dim=0)

        # concate real data
        if ISP:
            # Generate input random shift feature 
            pooling_time_ratio = 4
            shift_list = [random.randint(-64,64)*pooling_time_ratio for iter in range(cfg.batch_size)]
            freq_shift_list = [random.randint(-4,4) for iter in range(cfg.batch_size)]
            for k in range(batch_input.shape[0]):
                input_shift = torch.roll(batch_input[k], shift_list[k], dims=1)
                input_shift = torch.unsqueeze(input_shift, 0)
                input_freq_shift = torch.roll(batch_input[k], freq_shift_list[k], dims=2)
                input_freq_shift = torch.unsqueeze(input_freq_shift, 0)

                ema_input_shift = torch.roll(ema_batch_input[k], shift_list[k], dims=1)
                ema_input_shift = torch.unsqueeze(ema_input_shift, 0)
                ema_input_freq_shift = torch.roll(ema_batch_input[k], freq_shift_list[k], dims=2)
                ema_input_freq_shift = torch.unsqueeze(ema_input_freq_shift, 0)

                syn_input_shift = torch.roll(syn_batch_input[k], shift_list[k], dims=1)
                syn_input_shift = torch.unsqueeze(syn_input_shift, 0)
                syn_input_freq_shift = torch.roll(syn_batch_input[k], freq_shift_list[k], dims=2)
                syn_input_freq_shift = torch.unsqueeze(syn_input_freq_shift, 0)

                if k==0:
                    batch_input_shift = input_shift
                    batch_input_freq_shift = input_freq_shift

                    ema_batch_input_shift = ema_input_shift
                    ema_batch_input_freq_shift = ema_input_freq_shift

                    syn_batch_input_shift = syn_input_shift
                    syn_batch_input_freq_shift = syn_input_freq_shift
                else:
                    batch_input_shift = torch.cat((batch_input_shift,input_shift), 0)
                    batch_input_freq_shift = torch.cat((batch_input_freq_shift,input_freq_shift), 0)

                    ema_batch_input_shift = torch.cat((ema_batch_input_shift,ema_input_shift), 0)
                    ema_batch_input_freq_shift = torch.cat((ema_batch_input_freq_shift,ema_input_freq_shift), 0)

                    syn_batch_input_shift = torch.cat((syn_batch_input_shift,syn_input_shift), 0)
                    syn_batch_input_freq_shift = torch.cat((syn_batch_input_freq_shift,syn_input_freq_shift), 0)

                
            batch_input_shift = to_cuda_if_available(batch_input_shift)
            batch_input_freq_shift = to_cuda_if_available(batch_input_freq_shift)

            ema_batch_input_shift = to_cuda_if_available(ema_batch_input_shift)
            ema_batch_input_freq_shift = to_cuda_if_available(ema_batch_input_freq_shift)

            syn_batch_input_shift = to_cuda_if_available(syn_batch_input_shift)
            syn_batch_input_freq_shift = to_cuda_if_available(syn_batch_input_freq_shift)
        
        
        
 
        global_step = c_epoch * len(syn_loader) + i
        niter = c_epoch * len(syn_loader) + i
        # rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup*len(syn_loader))
        rampup_value = ramps.sigmoid_rampdown(c_epoch, 30)

        adv_step = (c_epoch-start_epoch) * len(syn_loader) + i
        rampup_value_adv = ramps.exp_rampup(adv_step, cfg.n_epoch_rampup*len(syn_loader))

        if adjust_lr:
            # adjust_learning_rate(optimizer, rampup_value, optimizer_d=optimizer_d, optimizer_crnn=optimizer_crnn, c_epoch=c_epoch, rampup_value_adv=rampup_value_adv)
            adjust_learning_rate(optimizer, rampup_value, optimizer_d=None, optimizer_crnn=None, c_epoch=c_epoch, rampup_value_adv=rampup_value_adv)
        meters.update('lr', optimizer.param_groups[0]['lr'])

       
        batch_input, ema_batch_input, target_weak = to_cuda_if_available(batch_input, ema_batch_input, target_weak)
        syn_batch_input, syn_ema_batch_input, syn_target = to_cuda_if_available(syn_batch_input, syn_ema_batch_input, syn_target)

        
        # Outputs
        
        
        


        adv_w = 1 # weight of adversarial loss
        update_step = 1
        # output_dim = 4096
        
        batch_size = cfg.batch_size


        syn_weak_pred = model(syn_batch_input)
        weak_pred = model(batch_input)
        # syn_encoded_x, syn_d_input = model(syn_batch_input)
        # syn_strong_pred, syn_weak_pred = predictor(syn_encoded_x)

        # encoded_x, d_input = model(batch_input)
        # strong_pred, weak_pred = predictor(encoded_x)

            
    



            
        loss = None
        # Weak BCE Loss
        
        
        # if mask_weak is not None:
        # ======================================================================================================
        # FOR WEAK LABEL
        # ======================================================================================================
        syn_target_weak = syn_target.max(-2)[0]  # Take the max in the time axis
        weak_index = target_weak.shape[0] // 2
        weak_class_loss = class_criterion(syn_weak_pred, syn_target_weak) + class_criterion(weak_pred[:weak_index], target_weak[:weak_index])
        

        # PSEUDO LABELING
        # if ema_model is not None:
        #     weak_class_loss = weak_class_loss + class_criterion(weak_pred, target_pl)

        if i == 0:
            log.debug(f"target: {target.mean(-2)} \n"
                f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                f"tensor mean: {batch_input.mean()}")
        meters.update('weak_class_loss', weak_class_loss.item())
        
        loss = weak_class_loss
    
        
        # if discriminator is not None:
        #     loss += domain_loss
        
        writer.add_scalar('Loss', loss.item(), niter)
        writer.add_scalar('Weak class loss', weak_class_loss.item(), niter)
        # writer.add_scalar('Strong class loss', strong_class_loss.item(), niter)

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        
        # Compute gradient and do optimizer step
        # optimizer.zero_grad()
        optimizer.zero_grad()
        # optimizer_crnn.zero_grad()
        # optimizer_d.zero_grad()
        loss.backward()
        optimizer.step()
        # if discriminator:
            # optimizer_crnn.step()
            # optimizer_d.step()
        # if discriminator is not None:
        #     optimizer_d.step()

        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)
            update_ema_variables(predictor, ema_predictor, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss



if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=1
    torch.manual_seed(2023)
    np.random.seed(2023)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("BSED 2022")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")
    # pretrain or adaptation
    parser.add_argument("-stage", '--stage', type=str, default='pretrain',
                    help="Choose the training stage of ADDA. 'pretrain' or 'adaptation'.")
    # clip-level or frame-level
    parser.add_argument("-level", '--level', type=str, default='frame',
                    help="Choose the level to do DA. 'clip' or 'frame'.")
    # Use fpn
    parser.add_argument("-fpn", '--use_fpn', action="store_true",
                    help="Whether to use CRNN_fpn architecture.")
    # Use Meanteacher
    parser.add_argument("-mt", '--meanteacher', action="store_true",
                    help="Whether to use mean teacher method.")
    # Use ICT, SCT, Weakly psuedo-labeling
    parser.add_argument("-ISP", '--ISP', action="store_true",
                    help="Whether to use three semi-supervised learning strategies.")


    f_args = parser.parse_args()
    pprint(vars(f_args))

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    stage = f_args.stage
    meanteacher = f_args.meanteacher
    ISP = f_args.ISP
    if ISP:
        meanteacher = True

    # model_name = 'test_adaptation_FPN'# name your own model
    model_name = cfg.at_model_name # name your own model

    store_dir = os.path.join("stored_data", model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    start_epoch = 8
    if start_epoch == 0:
        writer = SummaryWriter(os.path.join(store_dir, "log"))
        os.makedirs(store_dir, exist_ok=True)
        os.makedirs(saved_model_dir, exist_ok=True)
        os.makedirs(saved_pred_dir, exist_ok=True)
    else:
        writer = SummaryWriter(os.path.join(store_dir, "log"), purge_step=start_epoch)

    n_channel = 1
    add_axis_conv = 0

    # Model taken from 2nd of dcase19 challenge: see Delphin-Poulat2019 in the results.
    n_layers = 7
    crnn_kwargs = {"n_in_channel": n_channel, "nclass": len(cfg.bird_list), "attention": True, "n_RNN_cell": 128,
                   "n_layers_RNN": 2,
                   "activation": "glu",
                   "dropout": 0.5,
                   "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                   "nb_filters": [16,  32,  64,  128,  128, 128, 128],
                   "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]}
    
    # discriminator_kwargs = {"input_dim": 8192, "dropout": 0.5} # weak cdan
    discriminator_kwargs = {"input_dim": 80128, "dropout": 0.5} # basic dan
    predictor_kwargs = {"nclass":len(cfg.bird_list), "attention":True, "n_RNN_cell":128}

    pooling_time_ratio = 4  # 2 * 2

    out_nb_frames_1s = cfg.sr / cfg.hop_size / pooling_time_ratio
    median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)
    # median_window = 8
    # logger.debug(f"median_window: {median_window}")
    # ##############
    # DATA
    # ##############
    # dataset = ENA_Dataset(base_feature_dir=cfg.feature_dir, compute_log=True)
    # dfs = get_dfs(dataset, reduced_number_of_data)

    # # Meta path for psds
    # durations_synth = get_durations_df(cfg.synthetic)
    many_hot_encoder = ManyHotEncoder(cfg.bird_list, n_frames=cfg.max_frames // cfg.pooling_time_ratio)
    encod_func = many_hot_encoder.encode_strong_df
    weak_encod_func = many_hot_encoder.encode_weak


    # transforms_scaler = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
    # train_scaler_dataset = ENA_Dataset_unlabeled(preprocess_dir=cfg.train_unlabeled_feature_dir, encod_func=weak_encod_func, transform=transforms_scaler, compute_log=True)
    # val_scaler_dataset = ENA_Dataset(preprocess_dir=cfg.val_feature_dir, encod_func=encod_func, transform=transforms_scaler, compute_log=True)
    # syn_scaler_dataset = SYN_Dataset(preprocess_dir=cfg.synth_feature_dir, encod_func=encod_func, transform=transforms_scaler, compute_log=True)


    scaler_args = []
    scaler = Scaler()

    # if cfg.only_syn == True:
    #     scaler.calculate_scaler(syn_scaler_dataset) 
    # else:
    #     # scaler.calculate_scaler(train_scaler_dataset) 
    #     scaler.calculate_scaler(ConcatDataset([train_scaler_dataset, syn_scaler_dataset])) 
    

    transforms_real = get_transforms(cfg.max_frames, None, add_axis_conv,
                            noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    transforms_syn = get_transforms(cfg.max_frames, None, add_axis_conv,
                            noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    

    real_unlabeled_dataset = ENA_Dataset_unlabeled(preprocess_dir=cfg.train_unlabeled_feature_dir, encod_func=weak_encod_func, transform=transforms_real, compute_log=True)
    real_weak_dataset = ENA_Dataset(preprocess_dir=cfg.train_weak_feature_dir, encod_func=encod_func, transform=transforms_real, compute_log=True)
    syn_dataset = SYN_Dataset(preprocess_dir=cfg.synth_feature_dir, encod_func=encod_func, transform=transforms_syn, compute_log=True)

    scaler_val = Scaler()
    # scaler_val.calculate_scaler(val_scaler_dataset) 
    transforms_valid = get_transforms(cfg.max_frames, None, add_axis_conv,
                                      noise_dict_params={"mean": 0., "snr": cfg.noise_snr})
    val_dataset = ENA_Dataset(preprocess_dir=cfg.val_feature_dir, encod_func=encod_func, transform=transforms_valid, compute_log=True)
    
    
    

    # if meanteacher == False:
    #     real_dataset = torch.utils.data.ConcatDataset([real_dataset, syn_dataset])
    #     # real_dataset = torch.utils.data.TensorDataset(real_dataset, syn_dataset)
    # else:
    #     real_dataset = real_dataset
    
    
    
    real_unlabeled_dataloader = DataLoader(real_unlabeled_dataset, batch_size=cfg.batch_size//2, shuffle=True)
    real_weak_dataloader = DataLoader(real_weak_dataset, batch_size=cfg.batch_size//2, shuffle=True)
    syn_dataloader = DataLoader(syn_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    model = Net_resnet(pretrained=True)

    # else:
    #     model = Net_resnet(pretrained=False)
    #     model_path = os.path.join(saved_model_dir, 'baseline_best')
    #     expe_state = torch.load(model_path, map_location="cpu")
    #     model.load_state_dict(expe_state["model"]["state_dict"])
    # pdb.set_trace()
    optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs)
    # bce_loss = nn.BCELoss()

    state = {
        'model': {"name": model.__class__.__name__,
                  'args': '',
                'state_dict': model.state_dict()},
        'optimizer': {"name": optim.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optim.state_dict()},
        "many_hot_encoder": many_hot_encoder.state_dict()
    }

    save_best_cb = SaveBest("sup")

    # ##############
    # Train
    # ##############
    current_loss_min = np.inf
    results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
    for epoch in range(start_epoch, cfg.n_epoch):
        # crnn.train()
        # if stage == 'adaptation':
        #     discriminator.train()
        # predictor.train()
        # # crnn, crnn_ema, predictor = to_cuda_if_available(crnn, crnn_ema, predictor)
        # if stage == 'adaptation':
        #     # crnn, discriminator, predictor = to_cuda_if_available(crnn, discriminator, predictor)
        #     crnn, discriminator, domain_adv, predictor = to_cuda_if_available(crnn, discriminator, domain_adv, predictor)
            
        # else:
        #     crnn, predictor = to_cuda_if_available(crnn, predictor)
        model.train()
        model = to_cuda_if_available(model)
        
        # if meanteacher:
        #     crnn_ema.train()
        #     predictor_ema.train()
        #     crnn_ema, predictor_ema = to_cuda_if_available(crnn_ema, predictor_ema)

        #     # loss_value = train(training_loader, crnn, optim, epoch, ema_model=crnn_ema, ema_predictor=predictor_ema,
        #     #                 mask_weak=weak_mask, mask_strong=strong_mask, adjust_lr=cfg.adjust_lr, predictor=predictor, discriminator=discriminator, optimizer_d=optim_d, optimizer_crnn=optim_crnn, ISP=ISP)            
        #     loss_value = train_mt(real_unlabeled_dataloader, real_weak_dataloader, syn_dataloader, crnn, optim, epoch, ema_model=crnn_ema, ema_predictor=predictor_ema,
        #                     mask_weak=None, mask_strong=None, adjust_lr=cfg.adjust_lr, predictor=predictor, discriminator=domain_adv, optimizer_d=None, optimizer_crnn=None, ISP=ISP)
        # else:
            #     loss_value = train(real_dataloader, crnn, optim, epoch,
            #                     mask_weak=None, mask_strong=None, adjust_lr=cfg.adjust_lr, predictor=predictor, discriminator=discriminator, optimizer_d=optim_d, optimizer_crnn=optim_crnn, ISP=ISP)
        loss_value = train_mt(real_unlabeled_dataloader, real_weak_dataloader, syn_dataloader, model, optim, epoch, ema_model=None, ema_predictor=None,
                            mask_weak=None, mask_strong=None, adjust_lr=cfg.adjust_lr, predictor=None, discriminator=None, optimizer_d=None, optimizer_crnn=None, ISP=ISP)

        # Validation
        model.eval()
        # predictor.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        saved_path_list = [os.path.join("./stored_data", model_name, "predictions", "result.csv")]
        # real_dataset = torch.utils.data.ConcatDataset([real_unlabeled_dataset, real_weak_dataset])
        # predictions, valid_synth, durations_synth = get_predictions(crnn, syn_dataloader, many_hot_encoder.decode_strong, pooling_time_ratio,median_window=median_window, save_predictions=saved_path_list, predictor=predictor)
        # Validation with synthetic data (dropping feature_filename for psds)
        # valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
        # ct_matrix, valid_synth_f1, psds_m_f1 = compute_metrics(predictions, valid_synth, durations_synth)
        # writer.add_scalar('Strong F1-score', valid_synth_f1, epoch)
        # Real validation data
        # validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
        # durations_validation = get_durations_df(cfg.validation, cfg.audio_validation_dir)
        # logger.info("\n ### Real validation metric ### \n")
        # if f_args.use_fpn:     
        #     valid_predictions, validation_labels_df, durations_validation = get_predictions(crnn, val_dataloader, many_hot_encoder.decode_strong,
        #                                     pooling_time_ratio, median_window=median_window, predictor=predictor, fpn=True)
        # else:
        #     valid_predictions, validation_labels_df, durations_validation = get_predictions(crnn, val_dataloader, many_hot_encoder.decode_strong,
        #                                     pooling_time_ratio, median_window=median_window, predictor=predictor)
        # ct_matrix, valid_real_f1, psds_real_f1 = compute_metrics(valid_predictions, validation_labels_df, durations_validation)
        # writer.add_scalar('Real Validation F1-score', valid_real_f1, epoch)
        print("cross-trigger confusion matrix")
        # print_ct_matrix(ct_matrix)
        # Evaluate weak
        # weak_metric = get_f_measure_by_class(crnn, len(cfg.classes), validation_dataloader_weak, predictor=predictor)
        weak_metric_valid = get_f_measure_by_class(model, len(cfg.bird_list), val_dataloader, trained=True)
        writer.add_scalar("Weak F1-score macro averaged", np.mean(weak_metric_valid), epoch)  

        # Update state
        state['model']['state_dict'] = model.state_dict()
        state['optimizer']['state_dict'] = optim.state_dict()
        state['epoch'] = epoch
        state['weak_metric'] = np.mean(weak_metric_valid)
        print('weak: ' + str(np.mean(weak_metric_valid)))


        # Callbacks
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        # if cfg.save_best:
        #     if save_best_cb.apply(valid_synth_f1):
        #         model_fname = os.path.join(saved_model_dir, "baseline_best")
        #         torch.save(state, model_fname)
        #     results.loc[epoch, "global_valid"] = valid_synth_f1
        # results.loc[epoch, "loss"] = loss_value.item()
        # results.loc[epoch, "valid_synth_f1"] = valid_synth_f1

        # if cfg.early_stopping:
        #     if early_stopping_call.apply(valid_synth_f1):
        #         logger.warn("EARLY STOPPING")
        #         break
        
        if loss_value.item() < current_loss_min:
            model_fname = os.path.join(saved_model_dir, "baseline_best")
            torch.save(state, model_fname)
            current_loss_min = loss_value.item()

        # results.loc[epoch, "global_valid"] = valid_real_f1
        results.loc[epoch, "loss"] = loss_value.item()
        # results.loc[epoch, "valid_synth_f1"] = valid_real_f1

        # if cfg.early_stopping:
        #     if early_stopping_call.apply(valid_real_f1):
        #         logger.warn("EARLY STOPPING")
        #         break


    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        model = _load_crnn(state)
        logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")
    else:
        logger.info("testing model of last epoch: {}".format(cfg.n_epoch))
    # results_df = pd.DataFrame(results).to_csv(os.path.join(saved_pred_dir, "results.tsv"),
    #                                           sep="\t", index=False, float_format="%.4f")
    # ##############
    # Validation
    # ##############
    # crnn.eval()
    # predictor.eval()
    # # transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)
    # predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")

    # # validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)
    # # validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    # # validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
    # # durations_validation = get_durations_df(cfg.validation, cfg.audio_validation_dir)
    # # Preds with only one value
    # valid_predictions = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
    #                                     pooling_time_ratio, median_window=median_window,
    #                                     save_predictions=predicitons_fname, predictor=predictor)
    # compute_metrics(valid_predictions, validation_labels_df, durations_validation)