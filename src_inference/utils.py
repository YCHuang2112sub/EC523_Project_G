import torch
from torch import nn
from torch.nn import functional as F

def fetch(path):
    return open(path, 'rb')

def parse_prompt(prompt):
    # return prompt and weight 1.0
    return prompt, float(1)

class MakeCutouts(nn.Module):
    '''
    Generates multiple, variably-sized cutouts from an input image tensor and resizes them to a specified size.
    Do augmentations + resize the image.
    Input:

    An image tensor with dimensions [batch_size, channels, height, width].
    Parameters for the number of cutouts (cutn), the size to which each cutout will be resized (cut_size), and a power value influencing the size distribution of the cutouts (cut_pow).

    Output:

    A tensor containing all the resized cutouts concatenated along the batch dimension, with dimensions [cutn * batch_size, channels, cut_size, cut_size].
    '''
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)
    
def spherical_dist_loss(x, y):
    '''
    Measures the angular distance between vectors on a unit sphere for similarity comparisons.
    Suitable for working in abstract feature space
    '''
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al.
    Total Variation Loss promotes image smoothness by penalizing variations between adjacent pixel values, 
    helping to reduce noise and artifacts in generated or processed images.
    """
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    """
    Pixel values of an image fall within a specified range (e.g., -1 to 1), 
    helping to prevent issues like saturation and clipping in image generation tasks
    """
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

def mse_loss_batch(input, target):
    """
    Compute the Mean Squared Error (MSE) loss between input and target images on a pixel-wise basis for a batch of images.
    """
    if input.shape != target.shape:
        raise ValueError("Input and target must have the same shape")

    # Instantiate the MSE loss function
    mse_loss_function = nn.MSELoss()

    # Compute MSE loss
    return mse_loss_function(input, target)