import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from skimage.transform import rescale, resize
from dataset import ADE20K, get_single_loader, getDataRoot
from itertools import zip_longest
import os
from torchvision import transforms
import pickle
from pca import pca, featureNormalize
from collections import Counter
data_root = os.path.join(os.path.dirname(__file__), '../data')
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from scipy import sparse


NUM_KMEANS_CLUSTERS = 100

YAATEH_DATA_ROOT = "/Users/yaatehr/Programs/spatial_LDA/data/seg_data/images/training"
BOX_DATA_ROOT = "/home/yaatehr/programs/datasets/seg_data/images/training"
PICKLE_SAVE_RUN = True
def get_matrix_path(edge_len):
    return os.path.join(data_root, "grayscale_img_matrix_%d.pkl" % edge_len)


def stack_images_rows_with_pad(dataset,edge_len):
    """
    If/when we use a transform this won't be necessary
    """
    path = get_matrix_path(edge_len)

    print("checking baseline path: \n" , path)
    if not os.path.exists(path):
        list_of_images = []
        label_list = []

        dataset = get_single_loader(dataset, batch_size=1, shuffle_dataset=True)
        bar = tqdm(total= len(dataset))

        for step, (img, label) in enumerate(dataset):
            # if step > 100 or step > len(dataset) - 1:
            #     break
            list_of_images.append(img.flatten())
            label_list.append(label)
            if step % 50 == 0:
                bar.update(50)
        maxlen = max([len(x) for x in list_of_images])
        out = np.vstack([np.concatenate([b, np.zeros(maxlen-len(b))]) for b in list_of_images])
        print(out.shape)
        out = (out, label_list)
        with open(path, 'wb') as f:
            pickle.dump(out, f)
        return out
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def resize_im_shape(img_shape, maxEdgeLen = 50):
    x,y = img_shape
    if x > y:
        maxEdge = x
    else:
        maxEdge = y

    scalefactor = maxEdgeLen/maxEdge
    remainderEdge = int(min(x,y) * scalefactor)
    if x > y:
        return maxEdgeLen, remainderEdge
    return remainderEdge, maxEdgeLen

def resize_im(im, edge_len):
    return resize(im, resize_im_shape(im.shape, maxEdgeLen=edge_len), anti_aliasing=False)
    

def createFeatureVectors(max_edge_len):
    cnt = Counter()
    grayscaleDataset = ADE20K(grayscale=True, root=getDataRoot(), transform=lambda x: resize_im(x, max_edge_len), useStringLabels=True, randomSeed=49)#, numLabelsLoaded=10)

    #select most commoon label strings from tuples of (label, count)
    mostCommonLabels =  list(map(lambda x: x[0], grayscaleDataset.counter.most_common(25)))
    grayscaleDataset.selectSubset(mostCommonLabels, normalizeWeights=True)
    print(len(grayscaleDataset.counter))
    print("resized image size is: ", grayscaleDataset.__getitem__(0)[0].shape)
    # print("dataset len is: ", len(grayscaleDataset.image_paths))
    print("stacking and flattening images")

    stacked_images, label_list = stack_images_rows_with_pad(grayscaleDataset, max_edge_len)
    # normalized_images = featureNormalize(stacked_images)[0]
    transformer = IncrementalPCA(batch_size=79)
    U = transformer.fit_transform(stacked_images)
    # U = transformer.predict(stacked_images)

    # U = pca(normalized_images)[0]
    kmeans = MiniBatchKMeans(n_clusters=len(grayscaleDataset.class_indices.keys()))
    print('fitting KMEANS')

    # kmeans.fit(U.shape[1])

    # print('stacking vectors KMEANS')


    # # for step, img in enumerate(stacked_images):
    # #     if step == 0:
    # #         vstack = img
    # #         continue
    # #     vstack = np.vstack((vstack, img))
    
    # # print(vstack)
    # print(stacked_images.shape)
    # prediction = kmeans.predict(stacked_images)
    # print(prediction)
    # path = os.path.join(data_root, "baseline_run_incremental_%d.pkl" % max_edge_len)
    # with open(path, "wb") as f:
    #     eval_tup = (prediction, label_list, kmeans, stacked_images.shape)
    #     pickle.dump(eval_tup, f)

# createFeatureVectors()

for i in range(20, 400, 20):
    createFeatureVectors(i)



# def evaluate_predictions(eval_tup_path):
#     with open(eval_tup_path, "rb") as f:
#         prediction, label_list, kmeans, vstackshape= pickle.load(f)

#     for pred_label in prediction
