"""A pre-processing script for cropping objects of interest in images to
their detected bounding boxes."""

# Native Python imports
import os

# External package imports
import numpy as np
import cv2 as cv

def crop_object(img, corner_points, out_dir):
    """Function for cropping a single object in a single image, and saves
    the cropped image into the out_dir output directory."""

def sort_objects_by_class(img_dir, label_dir, output_root_dir):
    """Main function for cropping our objects of interest into their respective
    classes."""