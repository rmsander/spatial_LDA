import json
import numpy as np
import os

F_JSON = os.path.join("..", "..", "datasets", "seg_data", "tree.json")
print("Tree located at: {}".format(F_JSON))

# Read json file to create dict of label --> root
with open(F_JSON) as json_file:
    data = json.load(json_file)
    json_file.close()

print("JSON KEYS: {}".format(json_file))