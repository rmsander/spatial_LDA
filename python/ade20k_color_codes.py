import json
import numpy as np
import os
from utils import make_inverted_labelmap

USE_BOX = True

def get_all_sublabels(node, i=0):
    # Base case
    print(node.keys())
    if "children" not in node:
        return node
    # Recurse and iterate through children
    else:
        for child in node['children']:
            return get_all_sublabels(child, i=i+1)
    if i > 0:
        return node

def load_json(F_JSON):
    with open(F_JSON, "rb") as json_file:
        tree = json.load(json_file)
        json_file.close()
    return tree

# Debugging flag
USE_BOX = True

def main():
    if USE_BOX:
        F_JSON = os.path.join("..", "..", "datasets", "seg_data", "tree.json")
    else:
        F_JSON = os.path.join("..", "..", "slda_outside_work", "tree.json")
    DEPTH = 8
    labelmap = make_inverted_labelmap(DEPTH,
                                                  path_to_hierarchy=F_JSON)
    print("Tree located at: {}".format(F_JSON))

if __name__=="__main__":
    main()
