from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import glob
import numpy as np

path = "../dataanalysis/scannerID_2.csv"


def read_data(path):
    """
    Reads the data from the csv file and returns a pandas dataframe
    """
    return pd.read_csv(path, sep="\t", encoding="iso-8859-1")


def get_data(path):
    """
    Returns a pandas dataframe
    """
    return read_data(path)


def get_data_as_list(path):
    """
    Returns a list of lists
    """
    return read_data(path).values.tolist()


def grab_voxel_size(subject):
    """
    Returns a list of voxel sizes
    """

    flair = nib.load(subject + "/FLAIR.nii.gz").header
    sx, sy, sz = flair.get_zooms()
    return sx, sy, sz


# boxplot
def boxplot(path):
    """
    Returns a boxplot
    """
    data = get_data(path)
    ll = []
    # path_data = '/mnt/HDD18TB/martinsr/alldata2'
    path_data = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/alldata"
    path_to_experiment_data = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMHSEG_0.5/outputs/2022-05-09/23-35-18/testdatasplit_edit.txt"
    indices_to_drop = []
    all_subjects = []

    ptest = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMHSEG_0.5/outputs/2022-05-09/23-35-18/testdatasplit_edit.txt"
    with open(ptest, "r") as f:
        all_subjects_in_experiment = eval(f.read())

    ptrain = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMHSEG_0.5/outputs/2022-05-09/23-35-18/traindatasplit.txt"
    with open(ptrain, "r") as f:
        all_subjects_in_experiment += eval(f.read())

    pval = "/mnt/CRAI-NAS/all/martinsr/test_monai/WMHSEG_0.5/outputs/2022-05-09/23-35-18/valdatasplit.txt"
    with open(pval, "r") as f:
        all_subjects_in_experiment += eval(f.read())

    print(len(all_subjects_in_experiment))

    all_subjects_from_file = [
        x["image"].split("/")[-2] for x in all_subjects_in_experiment
    ]
    all_subjects = glob.glob(path_data + "/*")
    all_subjects = [
        x for x in all_subjects if x.split("/")[-1] in all_subjects_from_file
    ]

    for i in range(0, len(data.iloc[:, 0])):
        subject_id = data.iloc[i, 0]
        for subject in all_subjects:
            if subject_id == subject.split("/")[-1]:
                lock = False
                print(subject_id, subject)
                break
            else:
                lock = True
        if lock:
            indices_to_drop.append(i)
    data.drop(index=indices_to_drop, inplace=True, axis=0)
    # print(data)
    # Add voxel size to column
    for i in range(0, len(data.index)):
        subject_id = data.iloc[i, 0]
        g = list(data.index)[i]
        for subject in all_subjects:
            if subject_id in subject:
                x, y, z = grab_voxel_size(subject)
                data.loc[g, "voxel_sizex"] = round(x, 2)
                data.loc[g, "voxel_sizey"] = round(y, 2)
                data.loc[g, "voxel_sizez"] = round(z, 2)

    cols = list(data.columns.to_list())
    all_sites = np.unique(data["InstitutionName"].tolist())
    data_sites = {}
    counted = 0
    for site in all_sites:
        all_from_site = data[data["InstitutionName"] == site]
        all_softwareversion = all_from_site["SoftwareVersions"].tolist()
        all_voxelsx = all_from_site["voxel_sizex"].tolist()
        all_voxelsy = all_from_site["voxel_sizey"].tolist()
        all_voxelsz = all_from_site["voxel_sizez"].tolist()
        all_magnetic = all_from_site["MagneticFieldStrength"].tolist()
        all_repition = all_from_site["RepetitionTime"].tolist()
        all_echo = all_from_site["EchoTime"].tolist()
        all_flip = all_from_site["FlipAngle"].tolist()
        all_scanners = all_from_site["ManufacturerModelName"].tolist()
        data_sites[site] = [
            {
                "voxel_size": {
                    "minx": min(all_voxelsx),
                    "maxx": max(all_voxelsx),
                    "miny": min(all_voxelsy),
                    "maxy": max(all_voxelsy),
                    "minz": min(all_voxelsz),
                    "maxz": max(all_voxelsz),
                },
                "MagneticFieldStrength": {
                    "min": min(all_magnetic),
                    "max": max(all_magnetic),
                },
                "RepetitionTime": {"min": min(all_repition), "max": max(all_repition)},
                "EchoTime": {"min": min(all_echo), "max": max(all_echo)},
                "FlipAngle": {"min": min(all_flip), "max": max(all_flip)},
                "Scanner": np.unique(all_scanners, return_counts=True),
                "Software": np.unique(all_softwareversion, return_counts=True),
                "TOTAL": len(all_from_site),
            }
        ]
        print(np.unique(all_scanners, return_counts=True))
        print(site, data_sites[site][0]["voxel_size"])
        name, count = np.unique(all_scanners, return_counts=True)
        counted += count
    print(counted)
    print(data_sites)
    print(pd.DataFrame(data_sites))
    with open("../dataanalysis/data_sites_val.txt", "w") as f:
        f.write(str(data_sites))

    # data.to_excel("../dataanalysis/output_val.xlsx")
    # print(data)
    # print(len(data))

    # data = data.assign(VoxelSize=ll)
    # data = data.dropna()
    # data = data.drop('FlipAngle', 1)

    # print(data)
    # g = sns.PairGrid(data)
    # g.map_diag(sns.histplot)
    # g.map_offdiag(sns.scatterplot)
    # plt.savefig('../dataanalysis/boxplot_metadata.pdf')


boxplot(path)
