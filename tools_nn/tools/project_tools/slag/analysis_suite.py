import matplotlib.pyplot as plt
import nibabel as nib
import glob
import os
import matplotlib.colors as colors
import numpy as np
import seaborn
import pandas as pd
from tqdm import tqdm

class AnalysisSuite:
    def __init__(self, path):
        self.path_images = glob.glob(path + "/imagesTr_FLAIR/*.nii.gz")

    def get_mosaic(self, image, label, size=3, filename="mosaic.png", start_slice=150):

        # make color map for labels
        cmap = colors.ListedColormap(["black", "red"])

        fig, ax = plt.subplots(size, size, figsize=(20, 20))
        for i in range(size):
            for j in range(size):
                ax[i, j].imshow(
                    np.rot90(image[:, :, start_slice + i * size + j], k=1),
                    cmap="gray",
                    interpolation="none",
                    vmin=0,
                    vmax=np.quantile(image, 0.995),
                )
                ax[i, j].imshow(
                    np.rot90(label[:, :, start_slice + i * size + j], k=1),
                    interpolation="none",
                    alpha=0.5,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                )
                ax[i, j].axis("off")

        if not os.path.exists("mosaics_slag"):
            os.makedirs("mosaics")
        plt.tight_layout()
        plt.savefig("./mosaics_slag/" + filename)
        plt.close()

    def get_volume(self, filename):
        data_nib = nib.load(filename)
        data = data_nib.get_fdata()
        resolution = data_nib.header.get_zooms()
        return data, resolution

    def get_mosaic_all(self, size=3):
        for i in range(0, len(self.path_images)):
            image, resolution = self.get_volume(self.path_images[i])
            label, resolution = self.get_volume(self.path_images[i].replace("imagesTr", "labelsTr"))
            id = self.path_images[i].split("/")[-1].split(".")[0]
            self.get_mosaic(image, label, size, filename="mosaic" + str(id) + ".png")
    
    def get_median(self, data):
        return np.median(data)
    
    def get_mean(self, data):
        return np.mean(data)

    def get_std(self, data):
        return np.std(data)
    
    def get_min(self, data):
        return np.min(data)
    
    def get_max(self, data):
        return np.max(data)
    
    def quantile(self, data, q):
        return np.quantile(data, q)

    def get_stats(self, filename, labels = False) -> None:
        data, resolution = self.get_volume(filename)
        id = filename.split("/")[-1].split(".")[0]
        if labels:
            self.patient_stats[id] = {
                "median": self.get_median(data),
                "quantile_0.05": self.quantile(data, 0.05),
                "quantile_0.95": self.quantile(data, 0.95),
                "mean": self.get_mean(data),
                "std": self.get_std(data),
                "min": self.get_min(data),
                "max": self.get_max(data),
                "total slag [mL]": np.sum(data)*np.prod(resolution)*0.001,
            }

        else:
            self.patient_stats[id] = {
                "median": self.get_median(data),
                "quantile_0.05": self.quantile(data, 0.05),
                "quantile_0.95": self.quantile(data, 0.95),
                "mean": self.get_mean(data),
                "std": self.get_std(data),
                "min": self.get_min(data),
                "max": self.get_max(data),
            }
    
    def get_stats_all(self, labels = False):
        self.patient_stats = {}
        if labels:
            for i in range(0, len(self.path_images)):
                self.get_stats(self.path_images[i].replace("imagesTr", "labelsTr"), labels = labels)
        else:
            for i in range(0, len(self.path_images)):
                self.get_stats(self.path_images[i])
        return self.patient_stats
    
    def save_stats_json(self, filename):
        import json
        # pretty json
        with open(filename, 'w') as fp:
            json.dump(self.patient_stats, fp, indent=4)
    
    def plot_pair_plot(self, filename, labels = False):
        seaborn.set_palette("mako")
        df = pd.DataFrame.from_dict(self.patient_stats, orient='index')
        if labels:
            # plot histogram of sum
            seaborn.histplot(df["total slag [mL]"])
            plt.savefig(filename)
            plt.close()
        else:
            seaborn.pairplot(df, diag_kind="kde", markers="+")
            plt.savefig(filename)
            plt.close()



if __name__ == "__main__":
    analysis_suite = AnalysisSuite(
        "/mnt/CRAI-NAS/all/martinsr/NNunet/data/slagprosjekt"
    )
    # analysis_suite.get_mosaic_all()
    stats = analysis_suite.get_stats_all(labels = True)
    analysis_suite.save_stats_json("stats_labels.json")
    analysis_suite.plot_pair_plot("pairplot_labels.png", labels = True)
