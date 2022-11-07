
import torchvision
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureType,
    RandAffined,
    CenterSpatialCrop,
    SpatialPad,
    AsDiscreted,
    Resized,
    NormalizeIntensityd,
)
import random
import numpy as np
from skimage.measure import shannon_entropy
import torch

class RemoveEntropy:
    def __init__(self, keys, level = 2):
        self.keys = keys
        self.level = level


    def __call__(self, image_dict):
        # Remove slices with lower entropy than a given level
        arr = np.array([])
        image = image_dict[self.keys[0]]
        label = image_dict[self.keys[1]]
        axis = image.shape[0]
        # c b h w ->(b c) h w
        for i in range(0, axis):
            t = shannon_entropy(image[i, :, :, :])
            arr = np.append(arr, t)
        
        image2 = image[arr > self.level, :, :, :]
        label2 = label[arr > self.level, :, :, :]
        image_dict[self.keys[0]] = image2
        image_dict[self.keys[1]] = label2

        return image_dict


class RemoveEigthSlices:
    def __init__(self, keys):
        self.keys = keys


    def __call__(self, image_dict):
        # Remove slices with little information

        image = image_dict[self.keys[0]]
        label = image_dict[self.keys[1]]

        get_remove_num = image.shape[3]//8


        # image = image[:, :, :, get_remove_num:-get_remove_num]
        # label = label[:, :, :, get_remove_num:-get_remove_num]

        image = image[:, :, :, get_remove_num:-get_remove_num]
        label = label[:, :, :, get_remove_num:-get_remove_num]


        image_dict[self.keys[0]] = image
        image_dict[self.keys[1]] = label

        return image_dict

class Probe:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, image_dict):
        # Remove slices with little information

        image = image_dict[self.keys[0]]
        label = image_dict[self.keys[1]]

        print(image.shape)

        return image_dict

class Flipd_custom:
    def __init__(self, keys, prob = 0.1):
        self.keys = keys
        self.prob = prob
        self.flippinghorizontal = torchvision.transforms.RandomHorizontalFlip(p=1)


    def __call__(self, image_dict):
        # Remove slices with lower entropy than a given level
        image = image_dict[self.keys[0]]
        label = image_dict[self.keys[1]]

        probflip_hor = random.random() < self.prob
        prob_rot_amount = np.random.randint(0, 5)
        if probflip_hor:
            image = self.flippinghorizontal(image)
            label = self.flippinghorizontal(label)
        image = torch.rot90(image, k = prob_rot_amount, dims = (-2, -1)) 
        label = torch.rot90(label, k = prob_rot_amount, dims = (-2, -1))

        image_dict[self.keys[0]] = image
        image_dict[self.keys[1]] = label

        return image_dict


# define transforms for image and labelmentation
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero="True", channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
        AsDiscreted(keys = ["label"], threshold = 0.89),
        EnsureTyped(keys=["image", "label"], data_type = 'numpy'),
    ]
)


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero="True", channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
        AsDiscreted(keys = ["label"], threshold = 0.89),
        EnsureTyped(keys=["image", "label"], data_type = 'numpy'),
    ]
)

load_fazekas = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Resized(keys=["image"], spatial_size = (32, 32, 32)),
        EnsureTyped(keys=["image"])
    ]
)

correct_input = Compose([EnsureType('tensor'), CenterSpatialCrop(roi_size = [-1,256,256]), EnsureType('numpy'), SpatialPad(spatial_size = [-1,256,256], mode = 'minimum'), EnsureType('tensor')])


post_trans = Compose([EnsureType(), AsDiscrete(threshold = 0.89)])
augmentation = Compose([EnsureType(), Flipd_custom(keys = ["image", "label"], prob=0.5), RandAffined(keys = ["image", "label"], prob=0.1, shear_range = [-0.087, 0.087], rotate_range = [-0.26, 0.26], padding_mode = 'border', scale_range = [0.9, 1.1])])

# entropy_trans = Compose([RemoveEntropy(keys = ["image", "label"], level = 2)])

# flip_transform = Compose([EnsureType(), Flipd_custom(keys = ["image", "label"], prob=1)])