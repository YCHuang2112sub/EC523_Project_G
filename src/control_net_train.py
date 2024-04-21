import sys
import os

current_script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.insert(0, parent_dir)

script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import diffusers
import importlib
importlib.reload(diffusers)
import warnings
from tqdm import tqdm
from IPython.utils import io
import importlib 
from lib import data as anime_data
import lib.controlnet_self as controlnet_self_file
from lib.controlnet_self import ControlNetModel_SELF as ControlNetModel
from lib.controlnet_self import MultiControlNetModel_SELF 
importlib.reload(anime_data)
importlib.reload(controlnet_self_file)
import argparse
import contextlib
import gc
import logging
import math
import random
import shutil
from pathlib import Path
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from lib.controlnet_self import ControlNetModel_SELF as ControlNetModel
from lib.controlnet_self import MultiControlNetModel_SELF
from diffusers import (
    AutoencoderKL,
    # ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import wandb
importlib.reload(anime_data)
from huggingface_hub import login

from control_net_config import get_args_list, parse_args


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, HUGGING_FACE_CACHE_DIR):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        cache_dir = HUGGING_FACE_CACHE_DIR,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

def get_logger_and_accelerator(args, logger):
    # Set up logging
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    repo_id = None
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    return accelerator, logger, repo_id

# Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def get_tokenizer(args, accelerator):
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    return tokenizer

def load_and_setting_models(args, accelerator, HUGGING_FACE_CACHE_DIR, logger):
    

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, HUGGING_FACE_CACHE_DIR)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant,
        cache_dir = HUGGING_FACE_CACHE_DIR,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant,
        cache_dir = HUGGING_FACE_CACHE_DIR,
    )

    return noise_scheduler, text_encoder, vae

def get_controlnet_unet(args, accelerator, HUGGING_FACE_CACHE_DIR, logger):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,
        cache_dir = HUGGING_FACE_CACHE_DIR,
    )
    
    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path,
                                                        cache_dir = HUGGING_FACE_CACHE_DIR,
                                                        )
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)


    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet, accelerator).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet, accelerator).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    return controlnet, unet

def collate_fn(examples):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = torch.stack([torch.tensor(example["scene_img"]) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    conditioning_pixel_values = torch.stack([torch.tensor(example["inpainting_img"]) for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()


    figures = []
    for example in examples:
        img = example["figure_img_list"]
        # print(img.shape)    
        if img.ndim == 4:
            figures.append(torch.tensor(img))
        elif img.ndim == 5:
            figures.append(torch.tensor(img[:,0,:,:,:]))
        else:
            raise ValueError("figure_img_list should have 4 or 5 dimensions")        
    # conditioning_pixel_values_02 = torch.stack([torch.tensor(example["figure_img_list"][:,0,:,:,:]) for example in examples])
    conditioning_pixel_values_02 = torch.cat(figures, dim=0)
    conditioning_pixel_values_02 = conditioning_pixel_values_02.to(memory_format=torch.contiguous_format).float()


    captions = [example["description"] for example in examples]
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    # input_ids = torch.stack([example["input_ids"] for example in examples])
    input_ids = inputs.input_ids

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "conditioning_pixel_values_02": conditioning_pixel_values_02,
        "input_ids": input_ids,
        "captions": captions
    }

def get_dataloader(args, tokenizer, accelerator, dataset, split_ratio=0.9):    
        
    len_dataset = len(dataset)
    len_train = int(split_ratio * len_dataset)
    len_eval = len_dataset - len_train

    dataset_train = torch.utils.data.Subset(dataset, range(0, len_train))
    dataset_eval = torch.utils.data.Subset(dataset, range(len_train, len_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        dataset_eval,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    return train_dataloader, eval_dataloader

def get_optimizer(args, controlnet):    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    return optimizer

def get_lr_scheduler(args, optimizer):
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    return lr_scheduler

def wrap_with_accelerator(args, train_dataloader, eval_dataloader, controlnet, optimizer, lr_scheduler,
                          vae, unet, text_encoder, accelerator, overrode_max_train_steps=False):
    

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    return controlnet, optimizer, lr_scheduler, train_dataloader, eval_dataloader, vae, unet, text_encoder, \
        weight_dtype, num_update_steps_per_epoch

def log_validation(
    eval_dataloader, vae, text_encoder, tokenizer, unet, controlnet, \
    args, accelerator, weight_dtype, step, is_final_validation=False
):
    controlnet.eval()
    vae.eval()
    text_encoder.eval()
    unet.eval()

    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        controlnet = ControlNetModel.from_pretrained(
            args.output_dir, torch_dtype=weight_dtype,
            cache_dir = HUGGING_FACE_CACHE_DIR,
            )

    assert isinstance(controlnet, MultiControlNetModel_SELF)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        # controlnet=[controlnet, controlnet],
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        cache_dir = HUGGING_FACE_CACHE_DIR,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    global images
    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    i_image = 0
    for batch in tqdm(eval_dataloader, desc="Validation", unit="batch"):
        B = batch["pixel_values"].shape[0]
        for i in range(B):
            # print(batch["captions"][i])
            # print(batch["conditioning_pixel_values"][i].shape, batch["conditioning_pixel_values_02"][i].shape)
    
            i_image += 1
            if i_image > args.num_validation_images:
                break
            with inference_ctx:
                images = pipeline(
                    batch["captions"][i],
                    # batch["scene_img"][i],
                    image=[batch["conditioning_pixel_values"][i][None, :, :, :], batch["conditioning_pixel_values_02"][i][None, :, :, :]], # inpainting_img, figure_img
                    # num_inference_steps=20,
                    num_inference_steps=50,
                    # num_inference_steps=1000,
                    generator=generator,
                     num_images_per_prompt=3,
                ).images

            import torchvision
            img_grid_gen = torchvision.utils.make_grid([torch.tensor(np.array(image)).permute(2,0,1)for image in images], nrow=5)
            img_grid_gen = img_grid_gen.permute(1,2,0)
 
            image_logs.append(
                {"validation_image": batch["pixel_values"][i].detach().cpu().data.permute(1, 2, 0), 
                 "cond_01": batch["conditioning_pixel_values"][i].detach().cpu().data.permute(1, 2, 0),
                 "cond_02": batch["conditioning_pixel_values_02"][i].detach().cpu().data.permute(1, 2, 0),
                 "gen_images": images, 
                 "validation_prompt": batch["captions"][i]}
            )

        if i_image > args.num_validation_images:
            break

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["gen_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                cond_01_image = log["cond_01"]
                cond_02_image = log["cond_02"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))
                formatted_images.append(np.asarray(cond_01_image))
                formatted_images.append(np.asarray(cond_02_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["gen_images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                cond_01_image = log["cond_01"]
                cond_02_image = log["cond_02"]
                
                # Assuming images are in the range [0, 1], scale to [0, 255] and convert to numpy arrays
                validation_image_np = ((validation_image.detach().cpu().numpy() + 0.5) * 255).astype(np.uint8)
                cond_01_image_np = ((cond_01_image.detach().cpu().numpy() + 0.5) * 255).astype(np.uint8)
                cond_02_image_np = ((cond_02_image.detach().cpu().numpy() + 0.5) * 255).astype(np.uint8)

                
                formatted_images.append(wandb.Image(validation_image_np, caption="Controlnet conditioning"))
                formatted_images.append(wandb.Image(cond_01_image_np, caption="Controlnet conditioning 01"))
                formatted_images.append(wandb.Image(cond_02_image_np, caption="Controlnet conditioning 02"))   

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs

def training_loop(args, controlnet, optimizer, lr_scheduler, \
                  train_dataloader, eval_dataloader, \
                  vae, unet, text_encoder, tokenizer, noise_scheduler, \
                  weight_dtype, accelerator, \
                  logger, repo_id, num_update_steps_per_epoch
                  ):
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    vae.eval()
    text_encoder.eval()
    unet.eval()
    controlnet.train()


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        print(f"resume_from_checkpoint")
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in tqdm(range(first_epoch, args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            # THE FOLLOWING CODE SHOUD NOT HAVE ANY OUTPUT OF CELL
            
            
            vae.eval()
            text_encoder.eval()
            unet.eval()
            controlnet.train()
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                # print(f"batch['pixel_values'].shape: {batch['pixel_values'].shape}")
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                controlnet_image_01 = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                controlnet_image_02 = batch["conditioning_pixel_values_02"].to(dtype=weight_dtype)

                # assert to be multi-controlnet
                assert isinstance(controlnet, MultiControlNetModel_SELF)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=[controlnet_image_01, controlnet_image_02],
                    return_dict=False,
                    conditioning_scale=[1.0]*2,
                )
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                    if global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            eval_dataloader,
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet, accelerator)
        controlnet.save_pretrained(args.output_dir)

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(\
                eval_dataloader=eval_dataloader,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    if os.path.exists('/projectnb/dl523/students/ychuang2/'):
        HUGGING_FACE_CACHE_DIR = "/projectnb/dl523/students/ychuang2/.cache/huggingface/diffusers"
    else:
        HUGGING_FACE_CACHE_DIR = "./cache"
        
    project_main_path = Path.cwd().parent
    assert project_main_path.name == 'EC523_Project_G'
    added_path = os.path.abspath(project_main_path.__str__())
    if added_path not in os.sys.path:
        os.sys.patsh.append(added_path) 
    
    PHASE3_SCENE_DESCRIPTION_FILE = "./DATASET/PROCESSING_RECORD_PHASE3_SCENE_DESCRIPTION_train.json"
    dataset_path = os.path.abspath(project_main_path) # adjust the path to the dataset
    
    os.environ["WANDB_API_KEY"] = "80af4069cf927ce8e884699d422b0ddc3d7d1359"
    os.environ["WANDB_NOTEBOOK_NAME"] = "test_training_ControlNet.ipynb"
    
    wandb.login()
    login()
    
    MAX_NUM_FIGURE=1

    BATCH_SIZE = 8
    DATSET_SHUFFLE = True
    
    data_path_dict, anime_figure_scene_dataset = anime_data.get_dataset(PHASE3_SCENE_DESCRIPTION_FILE, dataset_path=dataset_path, MAX_NUM_FIGURE=MAX_NUM_FIGURE)
    
    logger = get_logger(__name__)
    
    checkpointing_steps = 16000 // (BATCH_SIZE // 4) // 2
    validation_steps = checkpointing_steps 
    num_train_epochs = 70
    
    args_list = get_args_list(BATCH_SIZE, num_train_epochs, checkpointing_steps, validation_steps)
    args = parse_args(BATCH_SIZE, input_args = args_list)
    
    accelerator, logger, repo_id = get_logger_and_accelerator(args, logger)
    
    tokenizer = get_tokenizer(args, accelerator)
    controlnet, unet = get_controlnet_unet(args, accelerator, HUGGING_FACE_CACHE_DIR, logger)

    # to multi-controlnet, copy the same onw twice , 1 -> 2x
    controlnet = MultiControlNetModel_SELF([controlnet, controlnet])

    noise_scheduler, text_encoder, vae = load_and_setting_models(args, accelerator,  HUGGING_FACE_CACHE_DIR, logger)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()
    
    train_dataloader, eval_dataloader = get_dataloader(args, tokenizer, accelerator, anime_figure_scene_dataset)
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    optimizer = get_optimizer(args, controlnet)
    lr_scheduler = get_lr_scheduler(args, optimizer)
    
    coontrolnet, optimizer, lr_scheduler, train_dataloader, eval_dataloader, vae, unet, text_encoder, \
    weight_dtype, num_update_steps_per_epoch = \
        wrap_with_accelerator(args, train_dataloader, eval_dataloader, controlnet, optimizer, lr_scheduler, \
                                vae, unet, text_encoder, accelerator, overrode_max_train_steps=overrode_max_train_steps)
        
    
    
    with io.capture_output() as captured:
        training_loop(args, controlnet, optimizer, lr_scheduler, \
                  train_dataloader, eval_dataloader, \
                  vae, unet, text_encoder, tokenizer, noise_scheduler, \
                  weight_dtype, accelerator, \
                  logger, repo_id, num_update_steps_per_epoch
                  )
    
    
    
