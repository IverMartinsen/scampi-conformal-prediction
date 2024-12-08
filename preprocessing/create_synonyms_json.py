import json
import pandas as pd
import numpy as np
from Levenshtein import distance as levdist

path_to_csv = "/Users/ima029/Library/CloudStorage/OneDrive-UiTOffice365/Prosjekter/SCAMPI/labelled crops/labels.csv"

labels = pd.read_csv(path_to_csv, index_col=0)

first_names = [label.rsplit(" ")[0].lower() for label in labels["label"]]

sorted_first_names = sorted(list(set((first_names))))

###
### Create dictionary of first name synonyms
###
first_name_synonyms = {}
indices = []
for i, token1 in enumerate(sorted_first_names):
    if not i in indices:
        first_name_synonyms[token1] = [token1]
    for j, token2 in enumerate(sorted_first_names[i + 1 :]):
        if levdist(token1, token2) < 3:
            first_name_synonyms[token1].append(token2)
            indices.append(j + i + 1)

###
### Assert that all first names are recorded
###
num = 0
for key in first_name_synonyms.keys():
    num += len(first_name_synonyms[key])

assert num == len(sorted_first_names)


###
### Reference list of (some) correct first names
###
correct_first_names = [
    "achomosphaera",
    "glaphyrocysta",
    "heteraulacacysta",
    "hystrichokolpoma",
    "hystrichosphaeridium",
    "hystrichostrogylon",
    "pyxidiniopsis",
    "sequoiapollenites",
    "spiniferites",
]


###
### Replace misspelled keys with correct first names
###
old_keys = []
new_keys = []

for key in first_name_synonyms.keys():
    for latin_name in correct_first_names:
        if latin_name in first_name_synonyms[key]:
            old_keys.append(key)
            new_keys.append(latin_name)

for i in range(len(old_keys)):
    first_name_synonyms[new_keys[i]] = first_name_synonyms.pop(old_keys[i])

###
### Save dictionary as json file
###
with open("synonyms.json", "w") as file:
    json.dump(first_name_synonyms, file, indent=4, sort_keys=True)


###
### Load dict of synonyms add column of correctly spelled first names to the labels dataframe
###

with open("synonyms.json", "r") as file:
    first_name_synonyms = json.load(file)

firstnames = []

for label in labels["label"]:
    for key in first_name_synonyms.keys():
        if label.lower().rsplit(" ")[0] in first_name_synonyms[key]:
            firstnames.append(key)

new_dataframe = pd.DataFrame(
    {"filename": labels["file"], "annotation": labels["label"], "firstname": firstnames}
)

###
### Count the number of occurences for each firstname, and store result in dataframe
###

firstnames = list(first_name_synonyms.keys())
count = [0 for i in firstnames]

for i, key in enumerate(first_name_synonyms.keys()):
    for label in labels["label"]:
        if label.lower().rsplit(" ")[0] in first_name_synonyms[key]:
            count[i] += 1

firstnames = list(np.array(id)[np.argsort(count)[::-1]])
count = list(np.array(count)[np.argsort(count)[::-1]])


pd.DataFrame({"label": firstnames, "count": count})
