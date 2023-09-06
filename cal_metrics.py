from myssim import compare_ssim as ssim
import numpy as np
import cv2
import pdb
import argparse
import os
import lpips
def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--udc_type', type=str, default='P')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()
    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    ss = []
    pp = []
    ll = []
    source_path = opt.source
    GT_path = f'/home/raid/wj/datasets/UDC_data/Test/{opt.udc_type}oled/HQ/'
    print(GT_path)
    p_source= os.listdir(source_path)
    p_GT = os.listdir(GT_path)
    p_source.sort()
    p_GT.sort()
    # pdb.set_trace()

    with open(f'{opt.tag}.txt','w',newline='') as f:
        f.write('Name'+'    ' + 'PSNR' + '    ' + 'SSIM'+ '    ' + 'LPIPS' + '\n')
        for name in p_source:
            a = cv2.imread(os.path.join(source_path, name)).astype('float')/255.0
            b = cv2.imread(os.path.join(GT_path, name)).astype('float')/255.0
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(source_path, name)))
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(GT_path, name)))
            s = ssim(a, b, gaussian_weights=True, use_sample_covariance=False,
                                 multichannel=True)
            p = output_psnr_mse(a, b)
            l = loss_fn.forward(img0, img1)
            ss.append(s)
            pp.append(p)
            ll.append(l.item())
            f.write(name + ',' + f'{p}' + ',' + f'{s}' + ',' + f'{l.item()}' + '\n')
        f.write(f'Avg PSNR is {sum(pp)/len(pp)}, Avg SSIM is {sum(ss)/len(ss)}, Avg lpips is {sum(ll)/len(ll)}')
    print(f'{opt.tag} inference complete!')
