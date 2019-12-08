# External package imports
import numpy as np
import cv2 as cv

# Native Python imports
import os
import pickle


def main():
    dir_files = os.path.join("..", "..", "datasets", "seg_data",
                                 "images", "dataset1")
    dir_segmented = os.path.join("..", "..", "datasets", "seg_data",
                                 "images", "training")
    sub_folders = os.listdir(dir_files)
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
                print("UNIQ", unique_vals, "COUNTS", unique_counts)
                segimg2class[f_img] = {
                    rgb2classes[unique_vals[i]]: unique_counts[i] / num_pixels
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
