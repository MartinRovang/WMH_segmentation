

from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureType,
    CenterSpatialCrop,
    SpatialPad,
    AddChannel,
    NormalizeIntensityd,
)
import numpy as np
import numpy as np
import nibabel as nib
# 

class NormalizeBrain:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, image_dict):
        # Z-standardize the volume based on the brain intensity values
        image = image_dict[self.keys[0]]
        image[image < 0] = 0
        brain_mean  = np.mean(image)
        brain_std = np.std(image)

        image = (image - brain_mean)/brain_std

        image_dict[self.keys[0]] = image

        return image_dict


class check_if_odd:
    
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, image_dict):
        """Shave off one linesegment of the slice if uneven"""
        image = image_dict[self.keys[0]]

        ss = image.shape
        if ss[0] % 2 != 0:
            image = image[1:,:,:]
        if ss[1] % 2 != 0:
            image = image[:,1:,:]
        if ss[2] % 2 != 0:
            image = image[:,:,1:]

        image_dict[self.keys[0]] = image
        return image_dict


class LoadImage_custom:
    
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, image_dict):
        """Shave off one linesegment of the slice if uneven"""
        # image_path = image_dict[self.keys[0]]
        print(image_dict[self.keys])
        # image = nib.load(image_path).get_fdata()
        # image_dict[self.keys[0]] = image
        return image_dict

preprocess_trans = Compose(
        [
            LoadImaged(keys=["img"]),
            check_if_odd(keys=["img"]),
            AddChanneld(keys=["img"]),
            NormalizeIntensityd(keys="img", nonzero="True", channel_wise=True),
            EnsureTyped(keys=["img"]),
        ]
    )


post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

def postprocessing_volume_end(size):
    post_trans_volume_adjust = Compose([AddChannel(), CenterSpatialCrop(roi_size = [size[0],size[1],size[2]]), EnsureType('numpy'), SpatialPad(spatial_size = [size[0],size[1],size[2]], mode = 'minimum'), EnsureType('numpy')])
    return post_trans_volume_adjust 