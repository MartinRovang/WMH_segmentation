import matplotlib.pyplot as plt
import nibabel as nib
import glob


path = [
    "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Lymedisease_control/502a.nii.gz"
]
path_to_save = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/Projects/Biobank"

example_img = (
    "/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/imagesTr/D110001-2.nii.gz"
)

for file in path:
    # as close canonical
    img = nib.load(file)
    img = nib.as_closest_canonical(img)

    example = nib.load(example_img).get_fdata()
    # save

    # plot slice from all directions
    fig, ax = plt.subplots(2, 3, figsize=(15, 5))
    ax[0, 0].imshow(example[:, 128, :], cmap="gray")
    ax[0, 0].set_title("Sagittal")
    ax[0, 1].imshow(example[128, :, :], cmap="gray")
    ax[0, 1].set_title("Coronal")
    ax[0, 2].imshow(example[:, :, 128], cmap="gray")
    ax[0, 2].set_title("Axial")

    ax[1, 0].imshow(img.get_fdata()[:, 128, :], cmap="gray")
    ax[1, 0].set_title("Sagittal")
    ax[1, 1].imshow(img.get_fdata()[128, :, :], cmap="gray")
    ax[1, 1].set_title("Coronal")
    ax[1, 2].imshow(img.get_fdata()[:, :, 128], cmap="gray")
    ax[1, 2].set_title("Axial")

    plt.savefig("example.png")
    # exit()
    plt.close()

    # nib.save(img, path_to_save + "/" + file.split("/")[-1])
    # print("Saved: ", file.split("/")[-1])
    # exit()
