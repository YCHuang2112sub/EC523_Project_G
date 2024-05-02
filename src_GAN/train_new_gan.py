from pathlib import Path
import os 
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


project_main_path = Path.cwd().parent.parent
assert project_main_path.name == 'EC523_Project_G_old'
added_path = os.path.abspath(project_main_path.__str__())
if added_path not in os.sys.path:
    os.sys.path.append(added_path)  


#get json file for evluation and test
PHASE3_SCENE_DESCRIPTION_FILE_TRAIN = "./DATASET/PROCESSING_RECORD_PHASE3_SCENE_DESCRIPTION_train.json"

PHASE3_SCENE_DESCRIPTION_FILE_TEST = "./DATASET/PROCESSING_RECORD_PHASE3_SCENE_DESCRIPTION_test.json"
dataset_path = os.path.abspath(project_main_path) # adjust the path to the dataset
project_main_path = Path.cwd().parent





MAX_NUM_FIGURE = 5
BATCH_SIZE = 1
DATASET_SHUFFLE = True



anime_figure_scene_dataset_train = anime_data.get_dataset(PHASE3_SCENE_DESCRIPTION_FILE_TRAIN, dataset_path=dataset_path, MAX_NUM_FIGURE=MAX_NUM_FIGURE)

anime_figure_scene_dataset_test= anime_data.get_dataset(PHASE3_SCENE_DESCRIPTION_FILE_TEST, dataset_path=dataset_path, MAX_NUM_FIGURE=MAX_NUM_FIGURE)
print(anime_figure_scene_dataset_train[0].keys())


len_train = len(anime_figure_scene_dataset_train)
len_test = len(anime_figure_scene_dataset_test)
print(f"len_train: {len_train}, len_test: {len_test}")
dataset_train = torch.utils.data.Subset(anime_figure_scene_dataset_train, range(0, len_train))
dataset_test = torch.utils.data.Subset(anime_figure_scene_dataset_test, range(0,len_test))
print(f"dataset_train: {len(dataset_train)}, dataset_test: {len(dataset_test)}")






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