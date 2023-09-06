import pdb

import numpy as np
import torch
import os

from .xsmn_lite import XSMN_Lite

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
# from network.Teacher.Model import Model as Teacher
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
# from mscv.cnn import normal_init
from loss import get_default_loss
from misc_utils import timer
# from loss import DA_multi_dis, DiscLoss
# from models.retinaface import RetinaFace, load_model
import misc_utils as utils

import  torch.nn as nn

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.cleaner = XSMN_Lite
        self.g_optimizer = get_optimizer(opt, [self.cleaner, self.netA])
        # self.d_optimizer = get_optimizer(opt, self.discriminator)

        self.scheduler = get_scheduler(opt, self.g_optimizer)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)
        self.criterion = nn.L1Loss()
    def update(self, x, y, epoch):

        # L1 & SSIM loss
        cleaned = self.cleaner(x)
        loss_G = self.criterion(cleaned, y.permute(0, 3, 1, 2))
        self.avg_meters.update(
            {'loss_all': loss_G.item(), 'loss_G': loss_G.item()})

        self.g_optimizer.zero_grad()
        loss_G.backward()
        self.g_optimizer.step()

        return {'recovered': cleaned}
    # @timer(False)
    def forward(self, x):
        return self.cleaner(x)

    def inference(self, x, progress_idx=None):
        return super(Model, self).inference(x, progress_idx)

    def load(self, ckpt_path):
        load_dict = {
            'cleaner': self.cleaner,
        }

        if opt.resume:
            load_dict.update({
                'optimizer': self.g_optimizer,
                'scheduler': self.scheduler,
            })
            utils.color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)
        else:
            utils.color_print('Load checkpoint from %s.' % ckpt_path, 3)

        ckpt_info = load_checkpoint(load_dict, ckpt_path, map_location=opt.device)
        epoch = ckpt_info.get('epoch', 0)

        return epoch

    def save(self, which_epoch):
        save_filename = f'{which_epoch}_{opt.model}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {
            'cleaner': self.cleaner,
            'optimizer': self.g_optimizer,
            'scheduler': self.scheduler,
            'epoch': which_epoch
        }

        save_checkpoint(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)

    def add_mapping_hook(self):

        def get_activation(mem, name):
            def get_output_hook(module, input, output):
                mem[name + str(output.device)] = output

            return get_output_hook

        def add_hook(net, mem, mapping_layers):
            for n, m in net.named_modules():
                if n in mapping_layers:
                    self.mapping_hooks.append(
                        m.register_forward_hook(get_activation(mem, n)))

        add_hook(self.teacher, self.Tacts, self.mapping_layers)
        add_hook(self.cleaner, self.Sacts, self.mapping_layers)


    def remove_mapping_hook(self):
        for hook in self.mapping_hooks:
            hook.remove()
        self.mapping_hooks = []


    def cal_distill_loss(self):
        losses = []
        for i, n in enumerate(self.mapping_layers):
            Tact = self.Tacts[n + str(opt.device)]
            Sact = self.Sacts[n + str(opt.device)]
            loss = self.distill_criterion(Tact, Sact)
            losses.append(loss)
        return sum(losses)

