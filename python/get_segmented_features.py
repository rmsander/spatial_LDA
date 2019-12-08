# External package imports
import numpy as np
import cv2 as cv

# Native Python imports
import os
import pickle


def main():
    dir_segmented = os.path.join("..", "..", "datasets", "seg_data",
                                 "images", "dataset1")
    sub_folders = os.listdir(dir_segmented)
    segimg2class = {}
    for sub_folder in sub_folders:
        print("Letter is: {}".format(letter))
        # Get files for specific label
        labels = os.listdir(os.path.join(dir_segmented, sub_folder))
        for label in labels:
            segmented_files = os.listdir(os.path.join(dir_segmented, sub_folder,
                                                      label))
            for f_img in segmented_files:
                A = cv.imread(os.path.join(f_img, dir_segmented, sub_folder,
                                            label, f_img)
                M, N, _ = A.shape
                num_pixels = M * N
                unique_vals, unique_counts = np.unique(A, return_counts=True, axis=2)
                print("UNIQ", unique_vals, "COUNTS", unique_counts)
                segimg2class[f_img] = {
                    rgb2classes[unique_vals[i]]: unique_counts[i] / num_pixels
                    for i in range(unique_counts.shape)}

    # Pickle results
    output_fname = os.path.join("..", "data", "SEG_COUNTS.pkl")
    with open(output_fname, "wb") as f:
        pickle.dump(segimg2class, output_fname)
        f.close()
    print("Seg Img Dictionary pickled to {}".format(output_fname))
    return segimg2class


if __name__ == "__main__":
    main()
