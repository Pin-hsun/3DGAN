import os

import tifffile
import tifffile as tiff
import torch
import glob
import numpy as np

def to_8_bit(img):
    img = (np.array((img - img.min()) / (img.max() - img.min())) * 255).astype(np.uint8)
    return img

def dice_coef(y_true, y_pred, epsilon=1e-6):
    """Altered Sorensen–Dice coefficient with epsilon for smoothing."""
    y_true_flatten = np.asarray(y_true).astype(np.bool)
    y_pred_flatten = np.asarray(y_pred).astype(np.bool)

    if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
        return 1.0

    return (2. * np.sum(y_true_flatten * y_pred_flatten)) / \
        (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon)

paths = glob.glob('/media/ziyi/glory/OAIDataBase/womac4min0/raw3D/SagIwTSE_test/*')
prj = '0728_cyc_oai3d_coherant'
ep = '50'
crop_size = 384
# seg_model = torch.load('submodels/seg_atten.pth', map_location=torch.device('cpu')).eval()
model = torch.load('/media/ziyi/glory/logs_pin/womac4min0/raw3D/{}/checkpoints/net_g_model_epoch_{}.pth'.format(prj, ep),
                    map_location=torch.device('cpu'))

for i in range(2,5):
    name = paths[i].split('/')[-1].split('.')[0]
    x0 = tiff.imread(paths[i])
    os.makedirs('inference_seg/{}_ep{}/{}'.format(prj, ep, name), exist_ok=True)
    tiff.imwrite('inference_seg/{}_ep{}/{}/ori.tif'.format(prj, ep, name), x0)
    x0[x0 >= 800] = 800
    x0 = (x0 - x0.min()) / (x0.max() - x0.min()) #(23, 384, 384)
    x0 = (x0 - 0.5) * 2
    x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float().permute(0, 1, 3, 4, 2)  # (B, C, H, W, D) torch.Size([1, 1, 384, 384, 23])
    upsample = torch.nn.Upsample(size=(384, 384, x0.shape[4] * 8), mode='trilinear')
    x0 = upsample(x0)
    crop = int((384 - crop_size) / 2)
    x0 = x0[:, :, crop:384 - crop, crop:384 - crop, :]

    out = model(x0)['out0']
    out = out[0].permute(1, 0, 2, 3)[::8, :, :, :]
    tifffile.imsave('inference_seg/{}_ep{}/{}.tif'.format(prj, ep, name), to_8_bit(out.detach().numpy()))

    # img_a = x0[0].permute(3, 0, 1, 2)[::8, :, :, :]  # ori_sag
    # img_b = out[0].permute(3, 0, 1, 2)[::8, :, :, :] #out sag
    # img_b_cor = out[0].permute(1, 0, 2, 3)[::8, :, :, :] #(H, C, D, W)
    # img_b_axi = out[0].permute(2, 0, 1, 3)[::8, :, :, :] #(W, C, D, H)
    #
    # # tiff.imwrite('inference_seg/img_a.tif', img_a.numpy())
    # # tiff.imwrite('inference_seg/img_b_cor.tif', img_b_cor.detach().numpy())
    # # tiff.imwrite('inference_seg/img_b_axi.tif', img_b_axi.detach().numpy())
    #
    # pred_a = torch.argmax(seg_model(img_a), 1, True)
    # pred_b = torch.argmax(seg_model(img_b), 1, True)
    # # pred_b_cor = torch.argmax(seg_model(img_b_cor), 1, True)
    # # pred_b_axi = torch.argmax(seg_model(img_b_axi), 1, True)
    #
    # # all1 = np.concatenate([to_8_bit(img_a).numpy(), to_8_bit(pred_a).detach().numpy(),
    # #                        to_8_bit(img_b).detach().numpy(), to_8_bit(pred_b).detach().numpy()], 2)
    # all1 = np.concatenate([to_8_bit(img_a), to_8_bit(pred_a.detach()),
    #                        to_8_bit(img_b.detach()), to_8_bit(pred_b.detach())], 2)
    # all2 = np.concatenate([img_b_cor.detach().numpy(), img_b_axi.detach().numpy()], 2)
    #
    # dice = round(dice_coef(pred_a.detach().numpy(), pred_b.detach().numpy()), 3)
    # print(dice)
    #
    # tiff.imwrite('inference_seg/{}_ep{}/{}/input.tif'.format(prj, ep, name), x0[0, 0, :, :, :].detach().numpy())
    # tiff.imwrite('inference_seg/{}_ep{}/{}/out.tif'.format(prj, ep, name), out[0, 0, :, :, :].detach().numpy())
    # tiff.imwrite('inference_seg/{}_ep{}/{}/Saggital_seg_dice{}.tif'.format(prj, ep, name, str(dice)), all1)
    # tiff.imwrite('inference_seg/{}_ep{}/{}/Coronal_Axial_seg.tif'.format(prj, ep, name), all2)
