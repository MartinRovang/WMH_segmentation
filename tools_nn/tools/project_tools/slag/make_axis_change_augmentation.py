import monai
import nibabel as nib


# def test():
#     #  nib close to canonical
#     img = nib.load('FLAIR1.nii.gz')
#     img = nib.as_closest_canonical(img)
#     # save
#     nib.save(img, 'test_canonical.nii.gz')

# test()

# orietnation transform
def make_axis_change_augmentation(path, path_label, save_path):
    
    data = {'image': path, 'label': path_label}

    # random 90 degree rotation and flip
    compose = monai.transforms.Compose(
        [   monai.transforms.LoadImaged(keys=['image', 'label']),
            monai.transforms.AddChanneld(keys=['image', 'label']),
            monai.transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            monai.transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            monai.transforms.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
        ]
    )
    data = compose(data)
    # save with same header and affine
    img = nib.load(path)
    img = nib.Nifti1Image(data['image'][0], img.affine, img.header)
    nib.save(img, save_path)

if __name__ == '__main__':
    make_axis_change_augmentation('FLAIR1.nii.gz', 'FLAIR1_label.nii.gz', 'test.nii.gz')
