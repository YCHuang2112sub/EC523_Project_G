# EC523_Project_G


## ✨🌠⬇️ Datasets
[https://drive.google.com/drive/folders/1kAZTuUCdl9n1POjpV3j5HnWQ8rzlQO6O?usp=drive_link]{Dataset Link (Make sure you login with BU account)}.

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


## Control-Net ヾ(*´▽‘*)ﾉ
All code for Controlnet implementation can be found here /EC523_Project_G/src_controlnet. To train, simply run python3 control_net_train.py. For training with different configurations,
 please modify the function def get_args_list() in control_net_config.py 
```
args_list = ["--pretrained_model_name_or_path", "stabilityai/stable-diffusion-2-1-base",
    	"--output_dir", "model_out",
    	"--dataset_name","multimodalart/facesyntheticsspigacaptioned",
    	"--conditioning_image_column", "spiga_seg", \
    	"--image_column", "image", \
    	"--caption_column","image_caption", \
    	"--resolution=256", \
    	"--learning_rate", "1e-5", \
    	# "--validation_image", '"./face_landmarks1.png " "./face_landmarks2.png" "./face_landmarks3.png"', \
    	# "--validation_prompt", '"High-quality close-up dslr photo of man wearing a hat with trees in the background" "Girl smiling, professional dslr photograph, dark background, studio lights, high quality" "Portrait of a clown face, oil on canvas, bittersweet expression"', \
    	"--train_batch_size",f"{BATCH_SIZE}", \
    	"--num_train_epochs",f"{num_train_epochs}", \
    	"--tracker_project_name", "controlnet", \
    	# "--enable_xformers_memory_efficient_attention", \
    	# "--checkpointing_steps", "5000", \
    	# "--validation_steps", "5000", \
    	"--checkpointing_steps", f"{checkpointing_steps}", \
    	"--validation_steps", f"{validation_steps}", \
    	"--report_to", "wandb", \
    	# "--push_to_hub",
    	"--cache_dir", "cache", \
    	"--num_validation_images", "20", \
    	"--has_eval_result_output_to_terminal", 0,  #"True" or 0 \
    	"--controlnet_embedding_merging_mode", "Addition",
    	#Addition, Concatenation_01, Concatenation_02, Attention
    	]
```
The model will save the checkpoint based on args_list and save it in the same directory of src_controlnet. To run an evaluation of ControlNet, run code control_net_evaluation.ipynb, which will generate an image and store it in the output_image folder. You can modify the model checkpoint by simply changing code:
```
checkpoint = "checkpoint-72000"
```


## Inference  ͡[๏̯͡๏]
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

## Evaluation


## Credits

This training code for GAN method was created based on `https://github.com/maximkm/StyleGAN-anime`. We would like to thank Maxim Nikolaev for making the code for GAN models publicly available. If you use would like to use the code in this repository, please cite the original repository.
