"""
Train a DM model for image generation
"""
import os
import argparse

from med_dataset import load_data
from improved_diffusion import dist_util, logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    med_model_and_diffusion_defaults,
    med_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

def create_argparser():
    defaults = dict(
        imgsize=512, 
        dataroot="/data1/tylin/NCTtoCCT/slice",
        log_dir="/data1/tylin/CKPT/DDPM",
        condclass="Art",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=1000000,
        batchsize=4,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(med_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def load_med_data(data_dir, imgsize, batchsize, train=True, phase=None, deterministic=False):
    data = load_data(
        data_root=data_dir, 
        batchsize=batchsize, 
        imgsize=imgsize,
        train=train,
        phase=phase,
        deterministic=deterministic,
    )
    med_data = []
    for batch in data:
        imgid, nctimg, cctimg = batch["imgid"], batch["nctimg"], batch["cctimg"]
        # cond img => nctimg & original img => cctimg
        imgdict = {
            "id": imgid,
            "nct": nctimg,
            "cct": cctimg
        }
        
        med_data.append(imgdict)
    return med_data

def main():
    """
    loading model => loading dataloader => start training
    
    loading model => loading training dataset and validation dataset => model training => validation
    """
    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    log_dir = os.path.join(args.log_dir, args.condclass)
    logger.configure(dir=log_dir, condclass=args.condclass)
    logger.log("loading model...")
    model, diffusion = med_create_model_and_diffusion(
        **args_to_dict(args, med_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("loading dataloader...")
    traindata = load_med_data(
        args.dataroot, 
        args.imgsize, 
        args.batchsize,
        train=True,
        phase=args.condclass,
        deterministic=False
    )
    
    logger.log("start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        traindata=traindata,
        batch_size=args.batchsize,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()
    
if __name__ == "__main__":
    main()