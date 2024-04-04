import torch
import torch.nn.functional as F
from torch.autograd import grad
from utils import register

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Load CLIP model and processor



gen_losses = register.ClassRegistry()
disc_losses = register.ClassRegistry()


@disc_losses.add_to_registry("bce")
def binary_cross_entopy(logits_real, logits_fake):
    labels_real = torch.ones_like(logits_real)
    loss = F.binary_cross_entropy_with_logits(logits_real, labels_real)
    labels_fake = torch.zeros_like(logits_fake)
    loss += F.binary_cross_entropy_with_logits(logits_fake, labels_fake)
    return loss


@gen_losses.add_to_registry("st_bce")
def saturating_bce_loss(logits_fake):
    zeros_label = torch.zeros_like(logits_fake)
    loss = -F.binary_cross_entropy_with_logits(logits_fake, zeros_label)
    return loss


@disc_losses.add_to_registry("hinge")
def disc_hinge_loss(logits_real, logits_fake):
    loss = F.relu(1.0 - logits_real).mean()
    loss += F.relu(1.0 + logits_fake).mean()
    return loss


@gen_losses.add_to_registry("hinge")
def gen_hinge_loss(logits_fake):
    loss = -logits_fake.mean()
    return loss


@disc_losses.add_to_registry("wgan")
def disc_wgan_loss(logits_real, logits_fake):
    loss = -logits_real.mean() + logits_fake.mean()
    return loss


@gen_losses.add_to_registry("wgan")
def gen_wgan_loss(logits_fake):
    loss = -logits_fake.mean()
    return loss


def compute_gradient_penalty(D, real_samples, fake_samples):
    # Calculates the gradient penalty loss for WGAN GP
    # Random weight term for interpolation between real and fake samples
    alpha = torch.randn((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0).requires_grad_(False)
    # Get gradient w.r.t. interpolates
    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


@disc_losses.add_to_registry("wgan-gp")
def disc_wgan_gp_loss(D, real_imgs, fake_imgs, lambda_gp):
    # Gradient penalty
    gradient_penalty = compute_gradient_penalty(D, real_imgs.data, fake_imgs.data)
    # Real images
    real_validity = D(real_imgs)
    # Fake images
    fake_validity = D(fake_imgs.detach())
    # Adversarial loss
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
    return d_loss
    
    
@gen_losses.add_to_registry("wgan-gp")
def gen_wgan_gp_loss(logits_fake):
    loss = -logits_fake.mean()
    return loss


def r1loss(inputs, label=None):
    # non-saturating loss with R1 regularization
    l = -1 if label else 1
    return F.softplus(l*inputs).mean()

##NEW IMPLEMENTATION: adding L2 loss to measure similarity between fake images and ground truth images
def L2loss(fake_imgs,ground_truth):
    return torch.nn.functional.mse_loss(fake_imgs, ground_truth)


def get_clip_loss(clip_model, fake_imgs, ground_truth_imgs):
    # Initialize the CLIP processor
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
    fake_imgs = normalize(fake_imgs)
    ground_truth_imgs = normalize(ground_truth_imgs )
    # fake_imgs_min, fake_imgs_max = fake_imgs.min(), fake_imgs.max()
    # ground_truth_imgs_min, ground_truth_imgs_max = ground_truth_imgs.min(), ground_truth_imgs.max()
    fake_imgs = fake_imgs.clamp(0, 1)
    ground_truth_imgs = ground_truth_imgs.clamp(0, 1)

    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Preprocess images
    fake_imgs_preprocessed = processor(images=fake_imgs, return_tensors="pt", padding=True)
    ground_truth_imgs_preprocessed = processor(images=ground_truth_imgs, return_tensors="pt", padding=True)
    
  
    with torch.no_grad():
        fake_features = clip_model.get_image_features(**fake_imgs_preprocessed)
        ground_truth_features = clip_model.get_image_features(**ground_truth_imgs_preprocessed)
    
    fake_features = fake_features / fake_features.norm(dim=-1, keepdim=True)
    ground_truth_features = ground_truth_features / ground_truth_features.norm(dim=-1, keepdim=True)
    
    # cosine similarity
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(fake_features, ground_truth_features)
    print(f" similarity is {similarity}")
    
    # Since we want to maximize similarity, minimize negative similarity as loss
    clip_loss = -similarity.mean()
    print(f"clip loss is {clip_loss}")
    
    return clip_loss




@disc_losses.add_to_registry("r1")
def disc_r1_loss(D, real_imgs, fake_imgs, ground_truth,lambda_gp):

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    #clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    real_imgs.requires_grad = True
    real_outputs = D(real_imgs)
    d_real_loss = r1loss(real_outputs, True)
    # Reference >> https://github.com/rosinality/style-based-gan-pytorch/blob/a3d000e707b70d1a5fc277912dc9d7432d6e6069/train.py
    # little different with original DiracGAN
    grad_real = grad(outputs=real_outputs.sum(), inputs=real_imgs, create_graph=True)[0]
    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
    grad_penalty = 0.5*lambda_gp*grad_penalty
    ## ADDING L2 loss
    l2_loss=L2loss(fake_imgs, ground_truth)
    print(f"l2 loss calculated{l2_loss}")

    clip_loss=get_clip_loss(clip_model,fake_imgs, ground_truth)
    print(f" calculated clip loss{clip_loss}")
    D_x_loss = d_real_loss + grad_penalty
    
    fake_logits = D(fake_imgs)
    D_z_loss = r1loss(fake_logits, False)
    D_loss = D_x_loss + D_z_loss+0.5*l2_loss+clip_loss
    print("Dloss at dis r1_loss", D_loss)
    return D_loss
    
    
@gen_losses.add_to_registry("r1")
def gen_r1_loss(logits_fake):
    loss = r1loss(logits_fake, True)
    return loss
