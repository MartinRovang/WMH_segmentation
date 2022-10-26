import json
import glob
import numpy as np

# read from file
with open('data.json', 'r') as infile:
    all_data_info = json.load(infile)


path_to_data = glob.glob("/mnt/CRAI-NAS/all/martinsr/NNunet/data/wmh_unique/imagesTs/*.nii.gz")

current_data = {}
data_of_a_given_type = []
for path in path_to_data:
    if "cor" in path:
        # print(path)
        continue
    elif "sag" in path:
        # print(path)
        continue
    else:
        id_ = path.split("/")[-1].split(".")[0]
        if id_ in all_data_info.keys():
            current_data[id_] = all_data_info[id_]
            data_of_a_given_type.append(current_data[id_]["ManufacturerModelName"])

names, counts = np.unique(data_of_a_given_type, return_counts=True)

print(names, counts)
print(sum(counts))