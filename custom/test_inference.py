import pytest
import cv2
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from gfpgan.utils import GFPGANer

################################
# global variable
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


################################

def test_inference():
    # initialize with the clean model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=400,               # Tile size for background sampler, 0 for no tile during testing
        tile_pad=10,
        pre_pad=0,
        half=True)              # need to set False in CPU mode

    restorer = GFPGANer(
        model_path='/content/GFPGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=bg_upsampler)
    # test GFPGANv1Clean attribute
    assert isinstance(restorer.gfpgan, GFPGANv1Clean)

    # ------------------ test enhance ---------------- #
    img = cv2.imread('data/00000000.png', cv2.IMREAD_COLOR)
    print(img)
    #result = restorer.enhance(img, has_aligned=False, paste_back=True)
    #assert result[0][0].shape == (512, 512, 3)
    #assert result[1][0].shape == (512, 512, 3)
    #assert result[2].shape == (1024, 1024, 3)
