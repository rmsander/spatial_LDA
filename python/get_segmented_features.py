# External package imports
import numpy as np
import cv2 as cv

# Native Python imports
import os
import pickle

def make_rgb_label_dict(rgb_dir)
    rgb_to_class = {}
    color_files = os.listdir(rgb_dir)
    # Filter out any non-jpg images
    color_files = [color_files[i] for i in range(len(color_files)) if \
                   color_files[i].endswith("jpg")]
    for color_file in color_files:
        A = cv.imread(os.path.join(rgb_dir, color_file))
        val = np.unique(A)
        class_name = color_file.split(".")[0]
        rgb_to_class[val] = class_name

    # Pickle results
    f_out =  os.path.join("..", "data", "rgb2class.pkl")
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
    dir_rgb_codes = os.path.join("..", "..", "datasets", "colors150")

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
                unique_vals, unique_counts = np.unique(A, return_counts=True, axis=2)
                print("UNIQ", unique_vals.shape, "COUNTS", unique_counts.shape)
                segimg2class[f_img] = {
                    rgb2class[unique_vals[i]]: unique_counts[i] / num_pixels
                    for i in range(unique_counts.shape[0])}

    # Pickle results
    output_fname = os.path.join("..", "data", "SEG_COUNTS.pkl")
    with open(output_fname, "wb") as f:
        pickle.dump(segimg2class, output_fname)
        f.close()
    print("Seg Img Dictionary pickled to {}".format(output_fname))
    return segimg2class


if __name__ == "__main__":
    main()
