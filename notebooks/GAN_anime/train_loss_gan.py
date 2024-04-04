from pathlib import Path
import os 
# add ../ to path
# added_path = os.path.abspath(str((Path.cwd().parent.parent / "lib")).__str__())
# if added_path not in os.sys.path:
#     os.sys.path.append(added_path)  

project_main_path = Path.cwd().parent.parent
assert project_main_path.name == 'EC523_Project_G'
added_path = os.path.abspath(project_main_path.__str__())
if added_path not in os.sys.path:
    os.sys.path.append(added_path)  



PHASE3_SCENE_DESCRIPTION_FILE = "./DATASET/PROCESSING_RECORD_PHASE3_SCENE_DESCRIPTION.json"
dataset_path = os.path.abspath(project_main_path) # adjust the path to the dataset
project_main_path = Path.cwd().parent


from pathlib import Path
import os
import importlib
import torch
from transformers import AutoTokenizer, PretrainedConfig
from matplotlib import pyplot as plt
import argparse
from lib import data
import importlib 
from lib import data as anime_data
importlib.reload(anime_data)
import torch

import importlib
importlib.reload(anime_data)

from torch.utils.data import DataLoader
from torchvision import transforms


MAX_NUM_FIGURE = 5
BATCH_SIZE = 4
DATASET_SHUFFLE = True



anime_figure_scene_dataset = anime_data.get_dataset(PHASE3_SCENE_DESCRIPTION_FILE, dataset_path=dataset_path, MAX_NUM_FIGURE=MAX_NUM_FIGURE)
# print(len(anime_figure_scene_dataset))
#print(anime_figure_scene_dataset[0].keys())


# import torch
# from torch.utils.data import  DataLoader

len_dataset = len(anime_figure_scene_dataset)
len_train = int(0.9 * len_dataset)
len_test = len_dataset - len_train
#print(f"len_train: {len_train}, len_test: {len_test}")
dataset_train = torch.utils.data.Subset(anime_figure_scene_dataset, range(0, len_train))
dataset_test = torch.utils.data.Subset(anime_figure_scene_dataset, range(len_train, len_dataset))
# print(f"dataset_train: {len(dataset_train)}, dataset_test: {len(dataset_test)}")


dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=DATASET_SHUFFLE)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=DATASET_SHUFFLE)

anime_data.display_data(dataloader_train, num_of_batch=1)
anime_data.display_data(dataloader_test, num_of_batch=1)



from start import init_train
from utils.images import SaveImages

# Loading models and the latest weights without loading the dataset
Trainer = init_train("configs/StyleGAN_256.json",dataloader_train, dataloader_test)
Trainer.train_loop()
# Save 1000 randomly generated images to the img/256_2 folder
SaveImages(Trainer, dir='img/256_2', cnt=1000)