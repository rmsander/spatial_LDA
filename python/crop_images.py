"""A pre-processing script for cropping objects of interest in images to
their detected bounding boxes."""

# Native Python imports
import os

# External package imports
import numpy as np
import cv2 as cv
import csv

def crop_object(img, corner_points, out_dir):
    """Function for cropping a single object in a single image, and saves
    the cropped image into the out_dir output directory."""

    pass
def sort_objects_by_class(img_dir, label_dir, output_root_dir):
    """Main function for cropping our objects of interest into their respective
    classes."""
    pass


def parse_bounding_csv(path_to_csv):
    """Parses the boundary box csv file. Gets information such as image ID,
    label name, boundaries (in percentages). Returns a dictionary of imageId:[(label, boundary)]"""
    parsed = {}
    with open(path_to_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                image_id = row[0]
                boundary = {}
                boundary["xMin"] = row[4]
                boundary["xMax"] = row[5]
                boundary["yMin"] = row[6]
                boundary["yMax"] = row[7]
                label = row[2]
                if image_id not in parsed:
                    parsed[image_id] = [(label, boundary)]
                else:
                    lis = parsed[image_id]
                    lis.append((label, boundary))
                    parsed[image_id] = lis
                line_count += 1
        print(f'Processed {line_count} lines.')
    return parsed

def parse_label_to_class_names(path_to_csv):
    """Returns a dictionary of label:class name"""
    parsed = {}
    with open(path_to_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            parsed[row[0]] = row[1]
            line_count += 1
        print(f'Processed {line_count} lines.')
    return parsed
            
            
