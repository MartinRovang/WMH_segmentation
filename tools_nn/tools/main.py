import pandas as pd
import json

data = pd.read_csv("scannerID_2.csv", header = 0, error_bad_lines=False, delimiter = '\t',encoding = "ISO-8859-1")

# add all into json
json_data = {}

for index, row in data.iterrows():
    json_data[row['subject-ass']] = row.to_dict()
    # remove subject ass in row
    del json_data[row['subject-ass']]['subject-ass']


# # write to file
# with open('data.json', 'w') as outfile:
#     json.dump(json_data, outfile)

# read from file
with open('alldata.json', 'r') as infile:
    alldata = json.load(infile)

temp = {}
for j, id_ in enumerate(alldata["id"]):
    temp[id_] = {"resolution": [alldata["resolution_x"][j], alldata["resolution_y"][j], alldata["resolution_z"][j]]}

# add to json data
for key in temp:
    try:
        json_data[key]["resolution"] = temp[key]["resolution"]
    except KeyError:
        print("KeyError: ", key)

# write json with some spacing
with open('data.json', 'w') as outfile:
    json.dump(json_data, outfile, indent=4)

print(len(json_data))