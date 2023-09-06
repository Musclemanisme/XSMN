# encoding=utf-8

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ski_ssim

import torch
import dataloader.dataloaders as dl
from options import opt
from mscv.summary import write_loss, write_image
from mscv.image import tensor2im
import numpy as np
from PIL import Image
from utils import *
from myssim import compare_ssim as ssim
import lpips

import misc_utils as utils
import pdb
def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

loss_fn_vgg = lpips.LPIPS(net='vgg')  # Used to calculate LPIPS
def evaluate(model, dataloader, epoch, writer, logger, data_name='val'):

    save_root = os.path.join(opt.result_dir, opt.tag, str(epoch), data_name)

    utils.try_make_dir(save_root)


    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    ct_num = 0
    # print('Start testing ' + tag + '...')
    metrics_path = save_root + 'metrics.txt'
    with open(metrics_path,'w') as f:
        for i, sample in enumerate(dataloader):
            utils.progress_bar(i, len(dataloader), 'Eva... ')

            path = sample['path']
            with torch.no_grad():
                recovered = model(sample['input'].to(device=opt.device))

            if data_name == 'val':
                label = sample['label']
                #############################
                # cal lpips
                if (opt.lpips):

                    total_lpips += loss_fn_vgg.to(device=opt.device)(recovered,label.to(device=opt.device)).item()
                #############################

                label = tensor2im(label)
                recovered = tensor2im(recovered)
                # import pdb
                # pdb.set_trace()
                # total_psnr += psnr(recovered, label, data_range=255)
                # total_ssim += ski_ssim(recovered, label, data_range=255, multichannel=True)
                recovered_ = recovered.astype('float') / 255.0
                label_ = label.astype('float') / 255.0
                img_ssim = ssim(label_, recovered_, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
                img_psnr = output_psnr_mse(label_, recovered_)
                total_ssim += img_ssim
                total_psnr += img_psnr
                ct_num += 1
                save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
                Image.fromarray(recovered).save(save_dst)
                # write metrics to txt file
                f.write(utils.get_file_name(path[0])+'.png'+','+f'{img_psnr}'+','+f'{img_ssim}' + '\n')

            elif data_name == 'test':
                pass

            else:
                raise Exception('Unknown dataset name: %s.' % data_name)

            # 保存结果
            # save_dst = os.path.join(save_root, utils.get_file_name(path[0]) + '.png')
            # Image.fromarray(recovered).save(save_dst)

        if data_name == 'val':
            ave_psnr = total_psnr / float(ct_num)
            ave_ssim = total_ssim / float(ct_num)

            # write_loss(writer, f'val/{data_name}', 'psnr', total_psnr / float(ct_num), epochs)
            if (opt.lpips):
                ave_lpips = total_lpips / float(ct_num)
                logger.info(f'Eva({data_name}) epoch {epoch}, lpips: {ave_lpips}.')
            logger.info(f'Eva({data_name}) epoch {epoch}, psnr: {ave_psnr}.')
            logger.info(f'Eva({data_name}) epoch {epoch}, ssim: {ave_ssim}.')


            return f'{ave_psnr: .4f}', f'{ave_ssim: .4f}'
        else:
            return ''


if __name__ == '__main__':
    from options import opt
    from network import get_model
    import misc_utils as utils
    from mscv.summary import create_summary_writer

    if not opt.load:
        print('Usage: eval.py [--tag TAG] --load LOAD')
        raise_exception('eval.py: the following arguments are required: --load')

    Model = get_model(opt.model)
    model = Model(opt)
    model = model.to(device=opt.device)

    opt.which_epoch = model.load(opt.load)

    model.eval()

    log_root = os.path.join(opt.result_dir, opt.tag, str(opt.which_epoch))
    utils.try_make_dir(log_root)

    writer = create_summary_writer(log_root)

    logger = init_log(training=False)
    evaluate(model, dl.val_dataloader, opt.which_epoch, writer, logger, 'val')

