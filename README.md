# EC523_Project_G


## Datasets

## Getting Started
Commands to get started:
```
module load python3
module load pytorch
module load tensorflow
[ ! -d "env" ] && virtualenv --system-site-packages env
source env/bin/activate
pip install -r requirements.txt
```
## WandB setup
[https://docs.wandb.ai/](WandB) is used for our projects. WandB provides stores training data online so it can be used easily to monitor all training tasks.

To create an account go to the wandb website and install through `pip install wandb` (or `pip install -r requirements.txt` ).


## Control-Net

## Inference
All code for baseline method implementation can be found here `/EC523_Project_G/src_inference`
To train simply run  `python3 inference_loop.py`. For training with different configuration you should modify source code in 199 lines in inference_loop.py model_config.update() and following codes
```
model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000',  
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_checkpoint': False,
        'use_fp16': True,
        'use_scale_shift_norm': True,
    })
    clip_guidance_scale = 2000  # Controls how much the image should look like the prompt.
    tv_scale = 300              # Controls the smoothness of the final output.
    range_scale = 200            # Controls how far out of range RGB values are allowed to be.
    cutn = 16
    skip_timesteps = 200  # This needs to be between approx. 200 and 500 when using an init image.
                        # Higher values make the output look more like the init.
    init_scale = 500      # This enhances the effect of the init image, a good value is 1000.
    mse_guidance_scale = 0 # Controls how much the image should look like the ground true image.
    # log hyperparameters
    wandb.config.update({
        'clip_guidance_scale': clip_guidance_scale,
        'tv_scale': tv_scale,
        'range_scale': range_scale,
        'cutn': cutn,
        'skip_timesteps': skip_timesteps,
        'init_scale': init_scale,
        'mse_guidance_scale': mse_guidance_scale
    })
```
## GAN-training
All code for GAN implementation can be found here `/EC523_Project_G/src_GAN`
To train simply run  `python3 train_new_gan.py`. For training with different configuration and model simply modify the following line  in train_new_gan.py

`Trainer = init_train("configs/StyleGAN_256.json",dataloader_train, dataloader_test)`

The main modification to the StyleGAN model can be found in `/EC523_Project_G/src_GAN/losses.py` where we incorporated customized loss function to achieve the goal of our project.

We also modified `/EC523_Project_G/src_GAN/trainer.py` to fit the specific structure of our datasets.

For debugging purposes we also provided a ipynb file you can find it here `/EC523_Project_G/notebooks/src_GAN/training_gan.ipynb`. You can run both the training and evluation code through this notebook. 


We also support using scc resources for training GAN model. If you would like to train using scc's GPU simply submit job by running `/EC523_Project_G/notebooks/src_GAN/qsub job.sh`. 

##Evaluation


## Credits

This training code for GAN method was created based on `https://github.com/maximkm/StyleGAN-anime`. We would like to thank Maxim Nikolaev for making the code for GAN models publicly available. If you use would like to use the code in this repository, please cite the original repository.
