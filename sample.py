import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    med_model_and_diffusion_defaults,
    med_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from train import load_med_data
from med_dataset import save_img

def create_argparser():
    defaults = dict(
        imgsize=512,
        dataroot='/data1/tylin/NCTtoCCT/slice',
        ckptroot='/data1/tylin/CKPT/DDPM/Art/CondDM-Art-03-20-15-27/ckpt/EMA-best.pt',
        savepath='/data1/tylin/CKPT/DDPM/Art/CondDM-Art-03-20-15-27',
        batchsize=1,
        use_ddim=False,
        clip_denoised=True
    )
    defaults.update(med_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

# load validation data
args = create_argparser().parse_args()
validdata = load_med_data(
    args.dataroot, 
    args.imgsize, 
    batchsize=1, 
    train=False,
    phase="Art",
    deterministic=True
)

# creating model
dist_util.setup_dist()
model, diffusion = med_create_model_and_diffusion(
    **args_to_dict(args, med_model_and_diffusion_defaults().keys())
)
model.load_state_dict(
    dist_util.load_state_dict(args.ckptroot, map_location="cpu")
)
model.to(dist_util.dev())
model.eval()

# starting sample
for i in validdata:
    imgid, cond = i["id"], {"nct": i["nct"]}
    model_kwargs = {k: v.to(dist_util.dev()) for k, v in cond.items()}
    output = diffusion.p_sample_loop(
        model,
        (args.batchsize, 1, args.imgsize, args.imgsize),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
    )
    # save img
    imgpath = os.path.join(args.savepath, "image")
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)
    imgname = str(imgid[0]).split(".")[0] + ".png"
    save_img(tensor=output, save_path=imgpath, img_name=imgname)