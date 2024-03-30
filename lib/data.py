import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import datasets
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
        # print("ratio: ", ratio)
        image = F.interpolate(image[None], size=(int(H/ratio), int(W/ratio)), mode='bilinear', align_corners=False)[0]
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
    # def __init__(self, scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list, MAX_NUM_FIGURE=5, dataset_path="./", figure_transform_flag = False):
    def __init__(self, data, MAX_NUM_FIGURE=5, dataset_path="./", figure_transform_flag = False):
        
        scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list = \
            data["scene_path_list"], data["inpainting_image_path_list"], data["fiugre_path_list"], data["description_list"]
        
        assert len(scene_path_list) == len(inpainting_image_path_list)
        assert len(scene_path_list) == len(fiugre_path_list)
        assert len(scene_path_list) == len(description_list)
        
        x = zip(scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list)
        # xt = [[scene, inpainting, figure, description] for scene, inpainting, figure, description in x  
        #                                                 if len(figure) <= MAX_NUM_FIGURE]
        xt = x
        scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list = zip(*xt)

        self.data = {"scene_path_list": scene_path_list, 
                     "inpainting_image_path_list": inpainting_image_path_list, 
                     "figure_path_list": fiugre_path_list, 
                     "description_list": description_list
                     }
        # self.scene_path_list = scene_path_list
        # self.inpainting_image_path_list = inpainting_image_path_list
        # self.figure_path_list = fiugre_path_list
        self.MAX_NUM_FIGURE = MAX_NUM_FIGURE
        # self.description_list = description_list

        self.dataset_path = dataset_path
        self.figure_transform_flag = figure_transform_flag

        # self.image_transforms = torchvision.transforms.Compose(
        #     [
        #         # torchvision.transforms.Resize(args.resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        #         # torchvision.transforms.CenterCrop(args.resolution),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize([0.5], [0.5]),
        #     ]
        # )

    def __len__(self):
        return len(self.data["scene_path_list"])

    def get_data_by_key(self, k):
        return self.data[k]

    def update(self, k, v):
        assert len(v) == self.__len__(), f"Length of {k} ({len(v)}) should be equal to the length of the dataset ({self.__len__()})."
        self.data[k] = v

    def get_scene_img_from_path(self, scene_path):
        scene_img = Image.open(Path(self.dataset_path) / scene_path)
        scene_img = scene_img.convert("RGB")
        scene_img = scene_img.resize((256, 256))
        scene_img = np.array(scene_img).transpose(2, 0, 1) / 255.0 -0.5
        return scene_img
    
    def get_inpainting_img_from_path(self, inpainting_image_path_list):
        len_inpainting_image = len(inpainting_image_path_list)
        i = np.random.randint(0, len_inpainting_image)
        inpainting_image_path = inpainting_image_path_list[i]["path"]
        inpainting_img = Image.open(Path(self.dataset_path) / inpainting_image_path)
        inpainting_img = inpainting_img.convert("RGB")
        inpainting_img = inpainting_img.resize((256,256))
        inpainting_img = np.array(inpainting_img).transpose(2, 0, 1) / 255.0 -0.5
        return inpainting_img
    
    def get_figure_img_from_path(self, figure_path_list):
        figure_img_list = np.zeros((self.MAX_NUM_FIGURE, 3, 256, 256))
        mask_img_list = np.zeros((self.MAX_NUM_FIGURE, 256, 256), dtype=bool)
        len_figure = min(len(figure_path_list), self.MAX_NUM_FIGURE)
        for i_figure, path in enumerate(figure_path_list):
            # figure_img_path = path["img_path"].replace("mask_", "")
            figure_img_path = path["img_path"]
            figure_img = Image.open(Path(self.dataset_path) / figure_img_path)
            figure_img = figure_img.convert("RGB")
            figure_img = np.array(figure_img).transpose(2, 0, 1)

            mask_img_path = path["mask_path"]
            mask_img = Image.open(Path(self.dataset_path) / mask_img_path)
            mask_img = mask_img.convert("L")
            mask_img = mask_img.resize((256, 256))
            mask_img = np.array(mask_img)
            mask_img = mask_img > 100

            if self.figure_transform_flag == True:
                figure_transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=0),
                    torchvision.transforms.RandomHorizontalFlip(),
                ])
            else:
                figure_transform = None

            figure_img = _random_pad_to_size(figure_img, size=(256, 256), transform=figure_transform)
            figure_img_list[i_figure] = figure_img / 255.0 - 0.5

            mask_img_list[i_figure] = mask_img

            if (i_figure+1) == len_figure:
                break
            
        return figure_img_list, len_figure, mask_img_list
    
    def get_description(self, description_list):
        len_descriptions = len(description_list)
        i = np.random.randint(0, len_descriptions)
        description = description_list[i]["description"].split("ASSISTANT:")[1]
        return description
    
    def __getitem__(self, idx):
        # print(idx)
        # scene_path = self.data["scene_path_list"][idx]
        # inpainting_image_path_list = self.data["inpainting_image_path_list"][idx]
        # figure_path = self.data["figure_path_list"][idx]
        # descriptions = self.data["description_list"][idx]

        # print(scene_path)
        # print(self.dataset_path)
        # print(Path(self.dataset_path))
        
        sampled_data = {}
        for k, v in self.data.items():
            sampled_data[k] = v[idx]

        sampled_data["scene_img"] = self.get_scene_img_from_path(sampled_data["scene_path_list"])
        sampled_data["inpainting_img"] = self.get_inpainting_img_from_path(sampled_data["inpainting_image_path_list"])
        sampled_data["figure_img_list"], sampled_data["len_figure"], sampled_data["mask_img_list"] = self.get_figure_img_from_path(sampled_data["figure_path_list"])
        sampled_data["description"] = self.get_description(sampled_data["description_list"])

        # leave one out inpainting image start
        scene = sampled_data["scene_img"]
        background = sampled_data["inpainting_img"]
        i_figure = np.random.randint(0, sampled_data["len_figure"])
        figure = sampled_data["figure_img_list"][i_figure]
        mask = sampled_data["mask_img_list"][i_figure]
        from copy import deepcopy
        left_one_out_background_img = deepcopy(scene)
        left_one_out_background_img[:,mask] = background[:,mask]
        sampled_data["inpainting_img"] = left_one_out_background_img
        sampled_data["figure_img_list"] = figure[np.newaxis, :, :, :]
        sampled_data["mask_img_list"] = mask[np.newaxis, :, :]
        sampled_data["len_figure"] = 1
        assert(len(sampled_data["figure_img_list"].shape) == 4 
               and len(sampled_data["mask_img_list"].shape) == 3)
        # leave one out inpainting image end

        sampled_data.pop("scene_path_list")
        sampled_data.pop("inpainting_image_path_list")
        sampled_data.pop("figure_path_list")
        sampled_data.pop("description_list")

        # return self.scene_path_list[idx], self.image_path_list[idx], self.fiugre_path_list[idx], self.description_list[idx]
        # return scene_img, inpainting_img, figure_img_list, description, len_figure
        # return {"scene_img": scene_img, "inpainting_img": inpainting_img, "figure_img_list": figure_img_list, "description": description, "len_figure": len_figure}
        return sampled_data

def _get_data_path(PHASE3_SCENE_DESCRIPTION_FILE, dataset_path):
    file_path = Path(dataset_path) / PHASE3_SCENE_DESCRIPTION_FILE
    if not Path(file_path).exists():
        raise FileNotFoundError(f"PHASE3_SCENE_DESCRIPTION_FILE: {file_path} does not exist.")
    
    with open(file_path, "r") as f:
        phase3_record = json.load(f)
    
    scene_path_list = []
    inpainting_image_path_list = []
    fiugre_path_list = []
    description_list = []

    print("Start Loading Metadata...")
    for series_name in phase3_record["dataset_dict"].keys():
        print(f"series_name: {series_name}")
        for ep_name in phase3_record["dataset_dict"][series_name].keys():
            # print(f"\t ep_name: {ep_name}")
            ep = phase3_record["dataset_dict"][series_name][ep_name]
            for idx_scene, scene in enumerate(ep):
                # print(f"\t\t scene idx: {idx_scene}")
                scene_data = scene["scene_data"]
                
                scene_path_list.append(scene_data["image_path_scene"])
                inpainting_image_path_list.append(scene_data["image_path_inpainting"])
                fiugre_path_list.append(scene_data["image_path_cropped_figures_with_bbox_segmt"])
                description_list.append(scene_data["image_scene_llava_description"])
    print("Finish Loading Metadata...")            
    
    # return scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list
    return {
            "scene_path_list": scene_path_list, 
            "inpainting_image_path_list": inpainting_image_path_list, 
            "fiugre_path_list": fiugre_path_list, 
            "description_list": description_list
            }

def get_dataset(PHASE3_SCENE_DESCRIPTION_FILE, dataset_path="./", MAX_NUM_FIGURE=5):
    data_path_dict = _get_data_path(PHASE3_SCENE_DESCRIPTION_FILE, dataset_path=dataset_path)
    # scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list = \
    #     data_path_dict["scene_path_list"], data_path_dict["inpainting_image_path_list"], data_path_dict["fiugre_path_list"], data_path_dict["description_list"]
    figure_scene_dataset = FigureSceneDataset(
        # scene_path_list=scene_path_list,
        # inpainting_image_path_list=inpainting_image_path_list,
        # fiugre_path_list=fiugre_path_list,
        # description_list=description_list,
        data = data_path_dict,
        MAX_NUM_FIGURE=MAX_NUM_FIGURE,
        dataset_path=dataset_path
    )
    return figure_scene_dataset


# def get_dataloader(PHASE3_SCENE_DESCRIPTION_FILE, batch_size, dataset_path="./", MAX_NUM_FIGURE=5, shuffle=False):
#     data_path_dict = _get_data_path(PHASE3_SCENE_DESCRIPTION_FILE, dataset_path=dataset_path)
#     scene_path_list, inpainting_image_path_list, fiugre_path_list, description_list = \
#         data_path_dict["scene_path_list"], data_path_dict["inpainting_image_path_list"], data_path_dict["fiugre_path_list"], data_path_dict["description_list"]
#     figure_scene_dataset = FigureSceneDataset(
#         scene_path_list=scene_path_list,
#         inpainting_image_path_list=inpainting_image_path_list,
#         fiugre_path_list=fiugre_path_list,
#         description_list=description_list,
#         MAX_NUM_FIGURE=MAX_NUM_FIGURE,
#         dataset_path=dataset_path
#     )
#     figure_scene_dataloader = DataLoader(figure_scene_dataset, batch_size, shuffle)
    
#     return figure_scene_dataloader

def display_data(dataloader,num_of_batch):
    for i in range(num_of_batch):
        batch = next(iter(dataloader))
        scene_img, inpainting_img, figure_img_list, description, num_figure = \
            batch["scene_img"], batch["inpainting_img"], batch["figure_img_list"], batch["description"], batch["len_figure"]
        plt.figure()
        plt.imshow(scene_img[0].permute(1,2,0))

        plt.figure()
        plt.imshow(inpainting_img[0].permute(1,2,0))

        for i in range(num_figure[0]):
            plt.figure()
            plt.imshow(figure_img_list[0][i].permute(1,2,0))
        
        print(description[0])