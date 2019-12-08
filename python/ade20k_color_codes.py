import json
import numpy as np
import os
from utils import make_inverted_labelmap


def main():
    #F_JSON = os.path.join("..", "..", "datasets", "seg_data", "tree.json")
    F_JSON = os.path.join("..", "..", "slda_outside_work", "tree.json")
    DEPTH = 8
    print("Tree located at: {}".format(F_JSON))

    # Read json file to create dict of label --> root
    inv_map = make_inverted_labelmap(DEPTH, path_to_hierarchy=F_JSON)
    print(inv_map)

if __name__=="__main__":
    main()
