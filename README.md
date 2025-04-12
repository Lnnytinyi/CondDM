# Conditional Diffusion Model
 
 **CondDM** is designed for NCCT (non-contrasted CT slices) to CECT (Contrasted CT slices) translation.

## Data configuration

Pair images of NCCT and CECT images will automatically be normlization during dataloading. Simply set `--data_dir path/to/pair/images` to the training script.

```
--- Art (Artery phase)
    |--- NCT (Non-contrast CT images)
    |--- CCT (Contrasted CT images)
--- Ven (Venous phase)
    |--- NCT (Non-contrast CT images)
    |--- CCT (Contrasted CT images)
```

NCT images are setted as condition for DM model, all which are dumped into a directory with ".jpg", ".jpeg", or ".png" extensions.

## Training

Here are some changes we experiment with, and how to set them in the flags:

 * **Learned sigmas:** add `--learn_sigma True` to `MODEL_FLAGS`
 * **Cosine schedule:** change `--noise_schedule linear` to `--noise_schedule cosine`
 * **Importance-sampled VLB:** add `--use_kl True` to `DIFFUSION_FLAGS` and add `--schedule_sampler loss-second-moment` to  `TRAIN_FLAGS`.
 * **Class-conditional:** add `--class_cond True` to `MODEL_FLAGS`.

Having setup hyper-parameters, to train the DDPM model,

```bash
sh train.sh
```

If want to train in a distributed manner, run the same command with `mpiexec`:

```bash
mpiexec -n $NUM_GPUS python train.py --dataroot path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

The logs and saved models will be written to a logging directory determined in log_dir.

## Sampling

Setup path to the conditional images directory `dataroot`, the generated target images directory `savepath`, and checkpoint directory `ckptroot`.

```bash
sh sample.sh
```

## Models and Hyperparameters

This section includes model checkpoints and run flags for the main models in the paper.

Note that the batch sizes are specified for single-GPU training, even though most of these runs will not naturally fit on a single GPU. To address this, either set `--microbatch` to a small value (e.g. 4) to train on one GPU, or run with MPI and divide `--batch_size` by the number of GPUs.

## Acknowledgments
We would like to thank the following individuals and organizations for their contributions to this project:
1. Thank the Improved DDPM paper [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), and [Github Repository: improved-diffusion](https://github.com/openai/improved-diffusion).

2. Thank the DDIM paper [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502).

3. Thank the DDPM paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), and [Github Repository: denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
