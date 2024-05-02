# Borrowed from https://github.com/maximkm/StyleGAN-anime
# modified for EC523 project
from IPython.display import clear_output
from utils import register, images
from tqdm import tqdm
import numpy as np
import torch
import warnings
import wandb
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import os
trainers = register.ClassRegistry()


@trainers.add_to_registry(name="base")
class BaseGANTrainer:
    def __init__(self, conf, **kwargs):
        '''
        Input:
            conf: dictionary of training settings
            **kwargs: the dictionary must contain:
                "G": initialized generator
                "D": initialized discriminator
                "start_epoch": continuing training
                "dataloader": dataset loader
                "optim_G": generator optimizer
                "optim_D": discriminator optimizer
                "gen_loss": generator loss function
                "disc_loss":  discriminator loss function
                "z_dim": the dimension of the generator vector
                "device": contains a device type
        '''
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.conf = conf
        if self.conf['wandb_set']:
            wandb.login()
            wandb.init(
                project=self.conf['Generator'],
                config=self.conf
            )
            wandb.watch(self.G)
            wandb.watch(self.D)
          

    def logger(self, data):
        if self.conf['wandb_set']:
            wandb.log(data)
        
  
    def save_model(self, epoch):
        state = {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'start_epoch': epoch + 1,
        }
        torch.save(state, f'{self.conf["Weight_dir"]}/weight {epoch + 1}.pth')


    def generate_images(self, text_emb=None,cnt=1):
        # Sample noise as generator input
        #print(cnt,self.z_dim)
        cnt=int(cnt)
        z_dim = int(self.z_dim)
        #print("z dim ",z_dim)
        z = torch.randn((1, z_dim)) 
        #print("z created") 
        z=z.to(self.device)
        text_emb=text_emb.to(self.device)
        return self.G(z,text_emb)
        
    
    def train_disc(self, real_imgs, fake_imgs,ground_truth,text_emb):
        real_logits = self.D(real_imgs,text_emb)
        fake_logits = self.D(fake_imgs,text_emb)
        true_logits = self.D(ground_truth,text_emb)
        return self.disc_loss(real_logits, fake_logits,true_logits)
        
        
    def train_gen(self, fake_imgs,text_emb):
        logits_fake = self.D(fake_imgs,text_emb)
        return self.gen_loss(logits_fake)


    # NEW IMPLEMENTATION: this function since our description might exceed maximum context size for
    #CLIP so we reduce the decription to max length 77 ideally in complete sentences
    def clip_text_simple(text, token_limit=77, tokenizer=None):
      
        if tokenizer is None:
            tokenizer = lambda txt: txt.split()
        print(f" debugging clip text {text[:100]}")
        sentences = text.split('.')
        clipped_text = ""
        token_count = 0

        for sentence in sentences:
            sentence_tokens = len(tokenizer(sentence))


            if token_count + sentence_tokens <= token_limit:
                clipped_text += sentence + '.'
                token_count += sentence_tokens
            else:
                break

        return clipped_text.strip()


   # Wrote eval_loop which supports final evaluation task
    def eval_loop(self):
        self.G.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # No need to track gradients
            for epoch in range(self.conf['epochs']):
                bar = tqdm(self.evalloader)
                generated_images = []

                for i, real_img in enumerate(bar):
                    # Generate text embeddings as done in training
                    directory="/projectnb/dl523/students/ellywang/EC523_Project_G_old/notebooks/StyleGAN-anime/img/description"
                    file_path = os.path.join(directory, f"description_{i}.txt")
                    texts = real_img['description'][0]
                    with open(file_path, 'w') as file:
                      file.write(texts)
                    model_name = "openai/clip-vit-base-patch32"
                    model = CLIPModel.from_pretrained(model_name)
                    processor = CLIPProcessor.from_pretrained(model_name)
                    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

                 
                    text_emb = model.get_text_features(**inputs)
                    linear=nn.Linear(512,256)
                    text_emb=linear(text_emb ).to(self.device) 

        

                    # Generate images
                    ground_truth=real_img["scene_img"].float().to(self.device)
                    fake_imgs = self.generate_images(text_emb, real_img["inpainting_img"].size(0))
                    real_imgs =  real_img["inpainting_img"].float().to(self.device)
                    
             
                    
                    if i !=1000 :
                        with torch.no_grad():
                            self.G.eval()
                            #saving images +text for evaluation
                            generated_p="/projectnb/dl523/students/ellywang/EC523_Project_G_old/notebooks/StyleGAN-anime/img/generated"
                            truth_p="/projectnb/dl523/students/ellywang/EC523_Project_G_old/notebooks/StyleGAN-anime/img/ground_truth"
                            scene_p="/projectnb/dl523/students/ellywang/EC523_Project_G_old/notebooks/StyleGAN-anime/img/scene"
                            Image_gen = images.TensorToImage(self.generate_images(text_emb).detach().cpu()[0], 0.5, 1)
                            Image_real= images.TensorToImage(real_imgs[0].cpu(), 0.5, 1)
                            Image_truth=images.TensorToImage(ground_truth[0].cpu(), 0.5, 1)
                            self.logger({"Ground Truth ":wandb.Image(Image_truth),"Random generated face": wandb.Image(Image_gen),"Input Scene ": wandb.Image(Image_real)})
                            image_gen=Image.fromarray(images.TensorToImage(self.generate_images(text_emb).detach().cpu()[0], 0.5, 1))
                            image_real=Image.fromarray(images.TensorToImage(real_imgs[0].cpu(), 0.5, 1))
                            image_truth=Image.fromarray(images.TensorToImage(ground_truth[0].cpu(), 0.5, 1))
                            file_truth = os.path.join(truth_p, f"ground_truth_{i}.png")
                            file_gen = os.path.join(generated_p, f"generated_{i}.png")
                            file_scene = os.path.join(scene_p, f"scene_{i}.png")
                            image_truth.save(file_truth) 
                            image_real.save(file_scene) 
                            image_gen.save(file_gen) 
                  
                
    
    def train_loop(self):
        warnings.filterwarnings("ignore")
        for epoch in range(self.start_epoch, self.conf['epochs']):
            #print(f"epoch {epoch}")
            bar = tqdm(self.dataloader)
            loss_G, loss_D = [], []
            # for our real_img in dataloader it consists of the following elements
            #dict_keys(['scene_img', 'inpainting_img', 'figure_img_list', 'len_figure', 'description'])
            # 1.'scene_img' -> ground truth of the anime scene
            # 2. 'description' -> text description of anime scene
            # 3. 'inpainting image' -> background of scene without characters
            # 4. "figure_img_list " -> consists of images of single/multiple characters in the scene_img
            # 5. 'len_figure' -> number of images in figure_image_list, equivalent to no. of characters
            for i, real_img in enumerate(bar):
                # print(f"real_img contains: {real_img.keys()}")
                # print(real_img["description"][0])
                description=real_img["description"][0]

                #Extract text embedding for conditioning
                model_name = "openai/clip-vit-base-patch32"
                model = CLIPModel.from_pretrained(model_name)
                processor = CLIPProcessor.from_pretrained(model_name)

                # Step 2: Prepare your text
                texts = real_img['description'][0]
                inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)

                # Step 3: Generate text embeddings
                text_emb = model.get_text_features(**inputs)
                linear=nn.Linear(512,256)
                text_emb=linear(text_emb ).to(self.device) 

      
   
                ground_truth=real_img["scene_img"].float().to(self.device)
                #print(f"keys are : {real_img.keys()} and shape is {real_img['pixel_values'].shape}")
                self.G.zero_grad()
                real_imgs =  real_img["inpainting_img"].float().to(self.device)
                
                # Generate a batch of images
                fake_imgs = self.generate_images(text_emb, real_imgs.size(0)).detach()

                # Update D network
                d_loss = self.train_disc(real_imgs, fake_imgs, ground_truth, text_emb)
                # print("d_loss is ",d_loss)
                
                self.D.zero_grad()
                d_loss.backward(retain_graph=True)
                self.optim_D.step()

                loss_D.append(d_loss.item())
                self.logger({"loss_D":d_loss.item()})

                if i % self.conf['UPD_FOR_GEN'] == 0:
                    # Generate a batch of images
                    fake_imgs = self.generate_images(text_emb,real_imgs.size(0))

                        # Update G network
                    g_loss = self.train_gen(fake_imgs,text_emb)
                        
                    self.G.zero_grad()
                    g_loss.backward(retain_graph=True)
                    self.optim_G.step()

                    loss_G.append(g_loss.item())
                    self.logger({"loss_G":g_loss.item()})

                # Output training stats
                if i % 5 == 0:
                    clear_output(wait=True)
                    with torch.no_grad():
                        self.G.eval()
                        Image = images.TensorToImage(self.generate_images(text_emb).detach().cpu()[0], 0.5, 1)
                        Image_real= images.TensorToImage(real_imgs[0].cpu(), 0.5, 1)
                        Image_truth=images.TensorToImage(ground_truth[0].cpu(), 0.5, 1)
                        self.logger({"Ground Truth ":wandb.Image(Image_truth),"Random generated face": wandb.Image(Image),"Input Scene ": wandb.Image(Image_real)})
                        self.G.train()

                clear_output(wait=True)        
                bar.set_description(f"Epoch {epoch + 1}/{self.conf['epochs']} D_loss: {round(loss_D[-1], 2)} G_loss: {round(loss_G[-1], 2)}")

            # Save model
            self.save_model(epoch)
            self.logger({"mean_loss_G":np.mean(loss_G), "mean_loss_D":np.mean(loss_D)})


@trainers.add_to_registry(name="gp")            
class GpGANTrainer(BaseGANTrainer):
    def __init__(self, conf, **kwargs):
        super().__init__(conf, **kwargs) 
        
        
    def train_disc(self, real_imgs, fake_imgs,ground_truth,text_emb):
        lambda_gp = self.conf["Loss_config"]["lambda_gp"]
        return self.disc_loss(self.D, real_imgs, fake_imgs, ground_truth,lambda_gp,text_emb)

