"""A pre-processing script for cropping objects of interest in images to
their detected bounding boxes."""

# Native Python imports
import os

# External package imports
import numpy as np
import cv2 as cv
import csv


def parse_bounding_csv(path_to_csv):
    """Parses the boundary box csv file. Gets information such as image ID,
    label name, boundaries (in percentages). Returns a dictionary of
    imageId:[(label, boundary)]"""
    parsed = {}
    with open(path_to_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print('Column names are {}'.format(row))
                line_count += 1
            else:
                image_id = row[0]
                if line_count == 1:
                    print(image_id)
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
        print('Processed {} lines'.format(line_count))
    return parsed


def parse_label_to_class_names(path_to_csv):
    """Returns a dictionary of label:class name"""
    parsed = {}
    with open(path_to_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            parsed[row[0][3:]] = row[1]
            # parsed[row[0]] = row[2]
            line_count += 1
        print("ROW: {}".format(row))
        print(f'Processed {line_count} lines.')
    return parsed


def crop_object(f_img, lb_pairs, out_root_dir, class_names):
    """Function for cropping a single object in a single image, and saves
    the cropped image into the out_dir output directory."""
    # Load image and get shape
    A = cv.imread(f_img)
    m, n, c = A.shape

    # Iterate over all labels in the image
    for lb_pair in lb_pairs:
        # Get label and boundary
        label, boundary = lb_pair

        # Convert min/max values from percentages to pixels
        xmin = int(n * float(boundary["xMin"]))
        xmax = int(n * float(boundary["xMax"]))
        ymin = int(m * float(boundary["yMin"]))
        ymax = int(m * float(boundary["yMax"]))

        # Crop image to object values over bounding box with all channels
        A_crop = A[ymin:ymax, xmin:xmax, :]

        # Get output directory using class label
        out_dir = os.path.join(out_root_dir, label[1:])

        # If directories (recursive) don't already exist, make them
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save image
        img_file = f_img.split("/")[-1]
        outfile = os.path.join(out_dir, img_file)
        cv.imwrite(outfile, A_crop)

def sort_objects_by_class(img_dir, csv_path, out_root_dir):
    """Main function for cropping our objects of interest into their respective
    classes."""
    # Get dictionary of parsed values
    parsed = parse_bounding_csv(csv_path)

    # Get image ids from parsed data
    ids = list(parsed.keys())
    counter = 0  # For printing

    # Get label to class dictionary
    class_names = parse_label_to_class_names(csv_path)
    
    files = os.listdir(img_dir) 

    # Iterate over all images via ids
    for id in ids:
        # Get filename
        fname = os.path.join(img_dir, id + ".jpg")
        # Extract all objects from file
        if fname.split("/")[-1] in files:
            crop_object(fname, parsed[id], out_root_dir, class_names)

        # Show progress
        if counter % 1000 == 0:
            print("Iterated over {} images".format(counter))
        counter += 1

def map_image_id_to_label(label_path, label):
    """Returns a dictionary of img_id:label"""
    mapped = {}
    img_files = os.listdir(label_path)
    for f in img_files:
        if f[-3:] != "jpg":
            continue
        mapped[f] = label
    print(len(mapped))
    return mapped

def main():
    img_dir = "/home/yaatehr/programs/datasets/google_open_image/train_00/"
    path_to_csv = "/home/yaatehr/programs/datasets/google_open_image/train" \
                  "-annotations-bbox.csv"
    out_root_dir = "/home/yaatehr/programs/spatial_LDA/data/cropped_test_0/"
    sort_objects_by_class(img_dir, path_to_csv, out_root_dir)

if __name__ == "__main__":
    main()
