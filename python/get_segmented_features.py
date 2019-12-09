# External package imports
import numpy as np
import cv2 as cv
from PIL import Image

# Native Python imports
import os
import pickle

def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    colors_tuple = [tuple(colors[i]) for i in range(colors.shape[0])]
    return colors_tuple, list(count)

def make_rgb_label_dict(rgb_dir):
    rgb_to_class = {(0, 0, 0):"NA"}
    color_files = os.listdir(rgb_dir)
    # Filter out any non-jpg images
    color_files = [color_files[i] for i in range(len(color_files)) if \
                   color_files[i].endswith("jpg")]
    for color_file in color_files:
        A = cv.imread(os.path.join(rgb_dir, color_file))
        colors, _ = unique_count_app(A)
        for color in colors:
            class_name = color_file.split(".")[0]
            rgb_to_class[color] = class_name

    # Pickle results
    f_out = os.path.join("..", "data", "rgb2class.pkl")
    with open(f_out, "wb") as f:
        pickle.dump(rgb_to_class, f)
        f.close()

    return rgb_to_class

def main():
    # Relevant directories
    dir_files = os.path.join("..", "..", "datasets", "seg_data",
                                 "images", "dataset1")
    dir_segmented = os.path.join("..", "..", "datasets", "seg_data",
                                 "images", "training")
    dir_rgb_codes = os.path.join("..", "..", "datasets", "seg_data", "color150")

    # Get RGB --> CLASS LABEL
    rgb2class = make_rgb_label_dict(dir_rgb_codes)
    print("RGB To Classes: \n {}".format(rgb2class))

    # Find folders to recursively iterate through
    sub_folders = os.listdir(dir_files)

    # Out data structure
    segimg2class = {}
    for sub_folder in sub_folders:
        print("Letter is: {}".format(sub_folder))
        # Get files for specific label
        labels = os.listdir(os.path.join(dir_files, sub_folder))
        for label in labels:
            print("LABEL IS: {}".format(label))
            segmented_files = os.listdir(os.path.join(dir_files, sub_folder,
                                                      label))
            for f_img in segmented_files:
                fname_split = f_img.split(".")
                seg_name = fname_split[0]+"_seg.png"
                print("FILE NAME IS: {}".format(seg_name))
                A = cv.imread(os.path.join(dir_segmented, sub_folder,
                                            label, seg_name))

                M, N, _ = A.shape
                num_pixels = M * N
                unique_vals, unique_counts = unique_count_app(A)
                print("UNIQUE KEYS: \n {}".format(len(list(rgb2class.keys(
                )))))
                segimg2class[f_img] = {}
                for i in range(len(unique_vals)):
                    key_of_key = tuple(list(unique_vals[i]))
                    print("KEY OF KEY: {}".format(key_of_key))
                    key = rgb2class[key_of_key]
                    value = unique_counts[i] / num_pixels
                    segimg2class[key] = value



    # Pickle results
    output_fname = os.path.join("..", "data", "SEG_COUNTS.pkl")
    with open(output_fname, "wb") as f:
        pickle.dump(segimg2class, output_fname)
        f.close()
    print("Seg Img Dictionary pickled to {}".format(output_fname))
    return segimg2class


if __name__ == "__main__":
    main()
