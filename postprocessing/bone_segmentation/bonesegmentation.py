import torch
from utils.bone_segmentation.unetclean import UNetClean
from collections import OrderedDict
from dotenv import load_dotenv
import os
load_dotenv('.env')


class BoneSegModel():
    def __init__(self):
        #net = UNetClean(4)
        #net.load_state_dict(torch.load('utils/bone_segmentation/clean_femur_tibia_cartilage.pth'))
        unet = UNetClean(output_ch=3)
        ckpt = torch.load(os.environ.get('model_seg'))
        state_dict = ckpt['state_dict']
        new_dict = OrderedDict()
        for k in list(state_dict.keys()):
            new_dict[k.split('net.')[-1]] = state_dict[k]
        unet.load_state_dict(new_dict)
        self.net = unet.cuda()


if __name__ == '__main__':
    from utils.bone_segmentation.bonesegmentation import BoneSegModel
    m = BoneSegModel()
    import torch
    from utils.make_config import load_config
    path = load_config('config.ini', 'Path')

    from dataloader.data import get_test_set

    root_path = path['dataset']  # "dataset/"
    test_set = get_test_set(root_path + 'painpickedgood', 'a_b', mode='test')