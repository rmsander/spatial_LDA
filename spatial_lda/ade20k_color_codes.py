import json
import os
from utils import make_inverted_labelmap

# Debugging flag
USE_BOX = True

def get_all_sublabels(node):
    sublabels = []
    queue = [node]
    while len(queue) > 0:
        n = queue.pop()
        sublabels.append(n["name"])
        try:
            childList = n["children"]
            for child in childList:
                queue.append(child)
        except:
            pass
    return sublabels

def load_json(F_JSON):
    with open(F_JSON, "rb") as json_file:
        tree = json.load(json_file)
        json_file.close()
    return tree

def depth(x):
    if type(x) is dict and x:
        return 1 + max(depth(x[a]) for a in x)
    if type(x) is list and x:
        return 1 + max(depth(a) for a in x)
    return 0

def find_ade150_nodes(tree, rgb_dict):
    queue = [root]
    while len(queue) > 0:

def main():
    if USE_BOX:
        F_JSON = os.path.join("..", "..", "datasets", "seg_data", "tree.json")
    else:
        F_JSON = os.path.join("..", "..", "slda_outside_work", "tree.json")
    print("DEPTH OF JSON FILE IS: {}".format(depth(load_json(F_JSON))))
    DEPTH = 10
    labelmap = make_inverted_labelmap(DEPTH,
                                                  path_to_hierarchy=F_JSON)
    print("Tree located at: {}".format(F_JSON))

if __name__=="__main__":
    main()
