import gc
import io
import math
import sys

from IPython import display
import lpips
from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

sys.path.append('../diffusion/CLIP')
sys.path.append('../diffusion/guided-diffusion')
sys.path.append('../lib')

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

import os
from pathlib import Path
import importlib 
import numpy as np

import data as anime_data
importlib.reload(anime_data)

import wandb
import json

from utils import fetch,parse_prompt,MakeCutouts,spherical_dist_loss,tv_loss,range_loss,mse_loss_batch

def train(model_config, device, anime_figure_scene_dataset, clip_guidance_scale, tv_scale, range_scale, cutn,skip_timesteps, init_scale, mse_guidance_scale,seed = 0, batch_size = 1):


    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load('256x256_diffusion_uncond.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    if model_config['use_fp16']:
        model.convert_to_fp16()

    clip_model = clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device)
    clip_size = clip_model.visual.input_resolution
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    #  LPIPS Model for Perceptual Similarity
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    def do_run_single(data, index, prompt_weight = 1, character_weight = 1):
        if seed is not None:
            torch.manual_seed(seed)

        make_cutouts = MakeCutouts(clip_size, cutn)
        side_x = side_y = model_config['image_size']

        target_embeds, weights = [], []

        # get text
        txt = data['description']
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt,truncate=True).to(device)).float())
        weights.append(prompt_weight)
        # save text to output_image/text
        with open(f'output_image/text/text_progress_{index}.txt', 'w') as f:
            f.write(txt)

        # get character image
        img = data["figure_img_list"][0].transpose(1, 2, 0)
        img = ((img + 0.5) * 255).astype(np.uint8)
        # save character image to output_image/character
        TF.to_pil_image(img).save(f'output_image/character/character_progress_{index}.png')
        
        img = Image.fromarray(img).convert('RGB')
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = clip_model.encode_image(normalize(batch)).float()
        target_embeds.append(embed)
        weights.extend([character_weight / cutn] * cutn)
        
        target_embeds = torch.cat(target_embeds)
        weights = torch.tensor(weights, device=device)
        if weights.sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        weights /= weights.sum().abs()

        # get inpainting_img image
        init = data["inpainting_img"].transpose(1, 2, 0)
        init = ((init + 0.5) * 255).astype(np.uint8)
        # save inpainting_img image to output_image/inpainting
        TF.to_pil_image(init).save(f'output_image/inpainting/inpainting_progress_{index}.png')
        
        init = Image.fromarray(init).convert('RGB')
        init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
        
        # get ground truth image
        ground_truth_img = data["scene_img"].transpose(1, 2, 0)
        ground_truth_img = ((ground_truth_img + 0.5) * 255).astype(np.uint8)
        # save ground truth image to output_image/ground_truth
        TF.to_pil_image(ground_truth_img).save(f'output_image/ground_truth/ground_truth_progress_{index}.png')
        
        
        ground_truth_img = Image.fromarray(ground_truth_img).convert('RGB')
        ground_truth_img = TF.to_tensor(ground_truth_img).to(device).unsqueeze(0).mul(2).sub(1)

        cur_t = None

        def cond_fn(x, t, out, y=None):
            n = x.shape[0]
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            image_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(image_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
            dists = dists.view([cutn, n, -1])
            losses = dists.mul(weights).sum(2).mean(0)
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out['pred_xstart'])
            
            # Compute the MSE loss with the ground truth
            mse_loss = mse_loss_batch(x_in, ground_truth_img.expand_as(x_in))
            
            loss = losses.sum() * clip_guidance_scale + tv_losses.sum() * tv_scale + range_losses.sum() * range_scale + mse_loss * mse_guidance_scale
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            return -torch.autograd.grad(loss, x)[0]

        if model_config['timestep_respacing'].startswith('ddim'):
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.p_sample_loop_progressive


        cur_t = diffusion.num_timesteps - skip_timesteps - 1

        samples = sample_fn(
            model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
            cond_fn_with_grad=True,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            # only save the last image
            if cur_t == -1:
                print("Saving image")
                for k, image in enumerate(sample['pred_xstart']):
                    filename = f'generated_progress_{index}.png'
                    # save to output_image/image
                    TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(f'output_image/generated/{filename}')
                    
        images_with_captions = [
            wandb.Image(f'output_image/character/character_progress_{index}.png', caption=f'Character {index}'),
            wandb.Image(f'output_image/inpainting/inpainting_progress_{index}.png', caption=f'Inpainting {index}'),
            wandb.Image(f'output_image/ground_truth/ground_truth_progress_{index}.png', caption=f'Ground Truth {index}'),
            wandb.Image(f'output_image/generated/generated_progress_{index}.png', caption=txt)
        ]
        
        wandb.log({'images': images_with_captions})
                        
    gc.collect()
    for i in range(len(anime_figure_scene_dataset)):
        do_run_single(anime_figure_scene_dataset[i], i)
    
                        
if __name__ == '__main__':
    json_path = "./config.json"
    with open(json_path, 'r') as config_file:
        config = json.load(config_file)
        wandb_api_key = config["WANDB"]["API"]
        entity = config["WANDB"]["Entity"]

    os.environ["WANDB_API_KEY"] = wandb_api_key
    wandb.init(project="inference_baseline", entity=entity)
    
    
    project_main_path = Path.cwd().parent
    assert project_main_path.name == 'EC523_Project_G'

    added_path = os.path.abspath(project_main_path.__str__())

    if added_path not in os.sys.path:
        os.sys.path.append(added_path)  
    PHASE3_SCENE_DESCRIPTION_FILE = "./DATASET/PROCESSING_RECORD_PHASE3_SCENE_DESCRIPTION_test.json"
    dataset_path = os.path.abspath(project_main_path) 

    MAX_NUM_FIGURE=1
    
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32, 16, 8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000',  # Modify this value to decrease the number of
                                    # timesteps.
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
    
    # log model config
    wandb.config.update(model_config)
    
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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    data_path_dict, anime_figure_scene_dataset = anime_data.get_dataset(PHASE3_SCENE_DESCRIPTION_FILE, dataset_path=dataset_path, MAX_NUM_FIGURE=MAX_NUM_FIGURE)
    train(model_config, device, anime_figure_scene_dataset, clip_guidance_scale, tv_scale, range_scale, cutn,skip_timesteps, init_scale, mse_guidance_scale)