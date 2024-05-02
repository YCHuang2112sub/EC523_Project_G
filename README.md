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