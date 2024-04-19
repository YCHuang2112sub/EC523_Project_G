import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torchvision
import matplotlib.pyplot as plt

def _random_pad_to_size(image, size=(256,256), transform=None):
    target_height, target_width = size
    
    image = torch.tensor(image)
    if transform is not None:
        image = transform(image)
    # Getting the original size
    C, H, W = image.shape
    ratio = max(H / size[0], W / size[1])
    if ratio > 1:
        print("ratio: ", ratio)
        #image = F.interpolate(image[None], size=(int(H/ratio), int(W/ratio)), mode='bilinear', align_corners=False)[0]
        image = F.interpolate(image[None].float(), size=(int(H/ratio), int(W/ratio)), mode='bilinear', align_corners=False)[0].byte()
    C, H, W = image.shape

    # Calculating the padding needed
    padding_height = max(target_height - H, 0)
    padding_width = max(target_width - W, 0)

    # Distributing the padding on both sides randomly
    top_pad = torch.randint(0, padding_height + 1, (1,)).item()
    bottom_pad = padding_height - top_pad

    left_pad = torch.randint(0, padding_width + 1, (1,)).item()
    right_pad = padding_width - left_pad

    # Applying the padding
    # Pad format is (left, right, top, bottom)
    padded_image = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad), 'constant', 0)

    # Verify the size
    assert padded_image.size(1) == 256 and padded_image.size(2) == 256, "Image was not properly padded."

    padded_image = padded_image.numpy()
    return padded_image

class FigureSceneDataset(Dataset):
    def __init__(self, scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list, MAX_NUM_FIGURE=5, dataset_path="./", figure_transform_flag = False):
        self.scene_path_list = scene_path_list
        self.inpainting_image_path_list = inpainting_image_path_list
        self.figure_path_list = fiugre_path_list
        self.MAX_NUM_FIGURE = MAX_NUM_FIGURE
        self.description_list = description_list

        self.dataset_path = dataset_path
        self.figure_transform_flag = figure_transform_flag

    def __len__(self):
        return len(self.scene_path_list)

    def __getitem__(self, idx):
        # print(idx)
        scene_path = self.scene_path_list[idx]
        inpainting_image_path_list = self.inpainting_image_path_list[idx]
        figure_path = self.figure_path_list[idx]
        descriptions = self.description_list[idx]

        # print(scene_path)
        # print(self.dataset_path)
        # print(Path(self.dataset_path))

        scene_img = Image.open(Path(self.dataset_path) / scene_path)
        scene_img = scene_img.convert("RGB")
        scene_img = scene_img.resize((256, 256))
        scene_img = np.array(scene_img) / 255.0

        len_inpainting_image = len(inpainting_image_path_list)
        i = np.random.randint(0, len_inpainting_image)
        inpainting_image_path = inpainting_image_path_list[i]["path"]
        inpainting_img = Image.open(Path(self.dataset_path) / inpainting_image_path)
        inpainting_img = inpainting_img.convert("RGB")
        inpainting_img = inpainting_img.resize((256,256))
        inpainting_img = np.array(inpainting_img) / 255.0

        figure_img_list = np.zeros((self.MAX_NUM_FIGURE, 3, 256, 256))
        len_figure = min(len(figure_path), self.MAX_NUM_FIGURE)
        for i_figure, path in enumerate(figure_path):
            figure_img_path = path["img_path"].replace("mask_", "")
            figure_img = Image.open(Path(self.dataset_path) / figure_img_path)
            figure_img = figure_img.convert("RGB")
            figure_img = np.array(figure_img).transpose(2, 0, 1)

            if self.figure_transform_flag == True:
                figure_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=0),
                    torchvision.transforms.RandomHorizontalFlip(),
                ])
            else:
                figure_transform = None

            figure_img = _random_pad_to_size(figure_img, size=(256, 256), transform=figure_transform)
            figure_img_list[i_figure] = figure_img / 255.0
            if (i_figure+1) == len_figure:
                break

        len_descriptions = len(descriptions)
        i = np.random.randint(0, len_descriptions)
        description = descriptions[i]["description"].split("ASSISTANT:")[1]

        # return self.scene_path_list[idx], self.image_path_list[idx], self.fiugre_path_list[idx], self.description_list[idx]
        return scene_img, inpainting_img, figure_img_list, description, len_figure
    
def _get_data_path(PHASE3_SCENE_DESCRIPTION_FILE):
    if not Path(PHASE3_SCENE_DESCRIPTION_FILE).exists():
        raise FileNotFoundError(f"PHASE3_SCENE_DESCRIPTION_FILE: {PHASE3_SCENE_DESCRIPTION_FILE} does not exist.")
    
    with open(PHASE3_SCENE_DESCRIPTION_FILE, "r") as f:
        phase3_record = json.load(f)
    
    scene_path_list = []
    inpainting_image_path_list = []
    fiugre_path_list = []
    description_list = []

    print("Start Loading Metadata...")
    for series_name in phase3_record["dataset_dict"].keys():
        print(f"series_name: {series_name}")
        for ep_name in phase3_record["dataset_dict"][series_name].keys():
            print(f"\t ep_name: {ep_name}")
            ep = phase3_record["dataset_dict"][series_name][ep_name]
            for idx_scene, scene in enumerate(ep):
                # print(f"\t\t scene idx: {idx_scene}")
                scene_data = scene["scene_data"]
                
                scene_path_list.append(scene_data["image_path_scene"])
                inpainting_image_path_list.append(scene_data["image_path_inpainting"])
                fiugre_path_list.append(scene_data["image_path_cropped_figures_with_bbox_segmt"])
                description_list.append(scene_data["image_scene_llava_description"])
    print("Finish Loading Metadata...")            
    
    return scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list

def get_dataloader(PHASE3_SCENE_DESCRIPTION_FILE, batch_size, dataset_path="./", MAX_NUM_FIGURE=5, shuffle=False):
    scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list = _get_data_path(PHASE3_SCENE_DESCRIPTION_FILE)
    figure_scene_dataset = FigureSceneDataset(
        scene_path_list=scene_path_list,
        inpainting_image_path_list=inpainting_image_path_list,
        fiugre_path_list=fiugre_path_list,
        description_list=description_list,
        MAX_NUM_FIGURE=MAX_NUM_FIGURE,
        dataset_path=dataset_path
    )
    figure_scene_dataloader = DataLoader(figure_scene_dataset, batch_size, shuffle)
    
    return figure_scene_dataloader

def display_data(dataloader,num_of_batch):
    for i in range(num_of_batch):
        scene_img, inpainting_img, figure_img_list, description, num_figure = next(iter(dataloader))
        plt.figure()
        plt.imshow(scene_img[0])

        plt.figure()
        plt.imshow(inpainting_img[0])

        for i in range(num_figure[0]):
            plt.figure()
            plt.imshow(figure_img_list[0][i].permute(1,2,0))
        
        print(description[0])