# Native Python imports
import os

# External package imports
import numpy as np
import cv2 as cv
import csv
import json
import pickle

import os.path
from os import path

LABEL_HIERARCHY_PATH = ""
DUMP_PATH = ""




if path.exists(DUMP_PATH):
    hierarchy = pickle.load(DUMP_PATH)
else:

    hierarchy_json = json.load(open(LABEL_HIERARCHY_PATH))
    hierarchy_json["json"]