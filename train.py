# encoding = utf-8
import logging
import os
import pdb
import time
import datetime
import csv
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import dataloader.dataloaders as dl
from network import get_model
from eval import evaluate

from options import opt

from utils import init_log, load_meta, save_meta, get_model_complexity_info
from mscv.summary import create_summary_writer, write_meters_loss

import misc_utils as utils
# from thop import profile
# from thop import clever_format
# from torchscan import summary



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)
# 初始化
with torch.no_grad():
    # 初始化路径
    save_root = os.path.join(opt.checkpoint_dir, opt.tag)
    log_root = os.path.join(opt.log_dir, opt.tag)

    utils.try_make_dir(save_root)
    utils.try_make_dir(log_root)

    # # Dataloader
    train_dataloader = dl.train_dataloader
    val_dataloader = dl.val_dataloader

    # Dataloader
    # train_dataloader = dl.disc_train_loader
    # val_dataloader = dl.disc_val_loader
    # 初始化日志
    logger = init_log(training=True)

    # 初始化实验结果记录表
    with open(file=os.path.join(log_root,'results.csv'), mode='a+',newline='') as f:
        csv_writer = csv.writer(f)
        if opt.resume:
            pass
        else:
            csv_writer.writerow(['Time','                Epoch','   PSNR','     SSIM'])
    print("CSV initialization complete!")


    # 初始化训练的meta信息
    meta = load_meta(new=True)
    save_meta(meta)

    # 初始化模型
    Model = get_model(opt.model)
    model = Model(opt)

    # 暂时还不支持多GPU
    # if len(opt.gpu_ids):
    #     model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    ###########使用循环算参数量##################
    # def count_param(model:nn.Module):
    #     param_count = 0
    #     for param in model.parameters():
    #         param_count += param.view(-1).size()[0]
    #     return param_count
    #
    # total_param = count_param(model)
    # print(f'Your params is {total_param/(1e6)}M')
    #############################################
    s_flops, s_params = get_model_complexity_info(model.cleaner, (3, 256, 256), print_per_layer_stat=False)
    print(f'Your model flops is {s_flops}, and params is {s_params}' + '\n')
    model = model.to(device=opt.device)
    # 加载预训练模型，恢复中断的训练
    if opt.load:
        load_epoch = model.load(opt.load)
        start_epoch = load_epoch + 1 if opt.resume else 1
    else:
        start_epoch = 1

    # 开始训练
    model.train()

    # 计算开始和总共的step
    print('Start training...')
    start_step = (start_epoch - 1) * len(train_dataloader)
    global_step = start_step
    total_steps = opt.epochs * len(train_dataloader)
    start = time.time()

    # Tensorboard初始化
    writer = create_summary_writer(log_root)

    start_time = time.time()

    # 在日志记录transforms
    logger.info('train_trasforms: ' +str(train_dataloader.dataset.transforms))
    logger.info('===========================================')
    if val_dataloader is not None:
        logger.info('val_trasforms: ' +str(val_dataloader.dataset.transforms))
    logger.info('===========================================')
    # logger.info(f'Your total params is {total_param/(1e6)}M')
    logger.info(f'Your total params is {s_params}')    # 不知道数量级对不对
    logger.info(f'Your total GFLOPS is {s_flops}')
    logger.info('===========================================')
try:
    # 训练循环
    for epoch in range(start_epoch, opt.epochs + 1):
        # import pdb
        # pdb.set_trace()
        for iteration, sample in enumerate(train_dataloader):
            global_step += 1
            print(sample['path'])
            # 计算剩余时间
            rate = (global_step - start_step) / (time.time() - start)
            remaining = (total_steps - global_step) / rate

            # update_return = model.update(sample['input'].to(device=opt.device), sample['label'].to(device=opt.device))
            update_return = model.update(sample['input'].to(device=opt.device), sample['label'].to(device=opt.device),epoch)


            #     logging.info(f'Your loss is NaN,the pic name is {sample["path"]}')
            # 获取当前学习率

            lr = model.get_lr()
            lr = lr if lr is not None else opt.lr

            # 显示进度条
            msg = f'lr:{round(lr, 6) : .6f} (loss) {str(model.avg_meters)} ETA: {utils.format_time(remaining)}'
            utils.progress_bar(iteration, len(train_dataloader), 'Epoch:%d' % epoch, msg)  #改成带总epochs的

            # 训练时每1000个step记录一下summary
            if global_step % 1000 == 0:
                write_meters_loss(writer, 'train', model.avg_meters, global_step)
                model.write_train_summary(update_return)

        # 每个epoch结束后的显示信息
        logger.info(f'Train epoch: {epoch}, lr: {round(lr, 6) : .6f}, (loss) ' + str(model.avg_meters))

        if epoch % opt.save_freq == 0 or epoch == opt.epochs:  # 最后一个epoch要保存一下
            model.save(epoch)

        # 训练中验证
        if epoch % opt.eval_freq == 0:

            model.eval()
            # eval_result = evaluate(model, val_dataloader, epoch, writer, logger)
            avg_psnr, avg_ssim = evaluate(model, val_dataloader, epoch, writer, logger)
            c_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # 写入此次val结果到csv中
            with open(file=os.path.join(log_root, 'results.csv'), mode='a+', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([c_time,'  '+ str(epoch),'   '+avg_psnr,'  '+avg_ssim])
            print(f"Epoch {epoch} results have been recorded! ")
            model.train()

        model.step_scheduler()

    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)

except Exception as e:

    if opt.tag != 'cache':
        with open('run_log.txt', 'a') as f:
            f.writelines('    Error: ' + str(e)[:120] + '\n')

    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)

    raise Exception('Error')  # 再引起一个异常，这样才能打印之前的错误信息

except:  # 其他异常，如键盘中断等
    meta = load_meta()
    meta[-1]['finishtime'] = utils.get_time_stamp()
    save_meta(meta)