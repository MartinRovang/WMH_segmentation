from itertools import count
import numpy as np
import os

# path_to_data = "/mnt/CRAI-NAS/all/martinsr/NNunet/data/Task01_BrainTumour/imagesTr"

# path_to_data = "/mnt/HDD16TB/martinsr/DatasetWMH211018_v2/alldata"

# read json file
path_to_data = "/mnt/CRAI-NAS/all/martinsr/NNunet/tools/dataset_original.json"
import json
# read
with open(path_to_data) as f:
    data = json.load(f)

train = data["training"]
test = data["test"]

all = train + test

def get_paths_to_data_and_shuffle(path_to_data):
    paths_to_data = [os.path.join(path_to_data, x) for x in os.listdir(path_to_data) if os.path.exists(os.path.join(path_to_data, x))]
    np.random.shuffle(paths_to_data)
    return paths_to_data


def get_unique_samples(paths_to_data):
    samples = []
    for path_to_data in paths_to_data:
        path_to_data = path_to_data["image"]
        sample_id = path_to_data.split("/")[-1].split("-")[0]
        samples.append(sample_id)
    unique_samples = np.unique(samples, return_counts=True)
    return unique_samples


# paths_to_data = get_paths_to_data_and_shuffle(path_to_data)
unique_samples, counts = get_unique_samples(all)

N = len(all)
M = len(unique_samples)

print((1 - (1 - M/N)) * 100, "% of the data is unique")

# # print all unique samples
# for i in range(len(unique_samples)):
#     if counts[i] == 3:
#         print(unique_samples[i])

# D220004K-2
# D440003K-1

def find_overlap_test_train():
    train = data["training"]
    test = data["test"]
    train_samples = get_unique_samples(train)[0]
    test_samples = get_unique_samples(test)[0]
    overlap = np.intersect1d(train_samples, test_samples)
    return overlap

overlap = find_overlap_test_train()

# percentage of test data
print(len(overlap)/len(test) * 100, "% of the test data is also in the training data")