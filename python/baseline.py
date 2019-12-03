import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans
from skimage.transform import rescale, resize
from dataset import ADE20K, get_single_loader
from itertools import zip_longest
import os
from torchvision import transforms
import pickle
from pca import pca, featureNormalize
from collections import Counter
data_root = os.path.join(os.path.dirname(__file__), '../data')


NUM_KMEANS_CLUSTERS = 100

YAATEH_DATA_ROOT = "/Users/yaatehr/Programs/spatial_LDA/data/seg_data/images/training"
BOX_DATA_ROOT = "/home/yaatehr/programs/datasets/seg_data/images/training"
PICKLE_SAVE_RUN = True
def get_matrix_path(edge_len):
    return os.path.join(data_root, "grayscale_img_matrix_%d.pkl" % edge_len)


def stack_images_rows_with_pad(list_of_images,edge_len):
    """
    If/when we use a transform this won't be necessary
    """
    maxlen = max([len(x) for x in list_of_images])
    out = np.vstack([np.concatenate([b, np.zeros(maxlen-len(b))]) for b in list_of_images])
    print(out.shape)
    if PICKLE_SAVE_RUN:
        path = get_matrix_path(edge_len)
        with open(path, 'wb') as f:
            pickle.dump(out, f)
    return out


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

    grayscaleDataset = ADE20K(grayscale=True, root=BOX_DATA_ROOT, transform=lambda x: resize_im(x, max_edge_len), useStringLabels=True, randomSeed=49, \
        labelSubset=['bathroom', 'game_room', 'dining_room', 'hotel_room', 'attic', 'outdoor'], normalizeWeights=True)

    dataset = get_single_loader(grayscaleDataset, batch_size=1, shuffle_dataset=True)
    print(grayscaleDataset.__getitem__(0)[0].shape)
    # print(grayscaleDataset.class_indices)
    print("dataset len: ", len(grayscaleDataset.image_paths))

    flattened_image_list = []
    label_list = []
    print("dataloader len: ", len(dataset))
    for step, (img, label) in enumerate(dataset):
        # if step > 100 or step > len(dataset) - 1:
        #     break
        flattened_image_list.append(img.flatten())
        cnt[label] +=1
        label_list.append(label)
    
    print(cnt)
    print(len(cnt))
    stacked_images = stack_images_rows_with_pad(flattened_image_list, max_edge_len)
    normalized_images = featureNormalize(stacked_images)[0]
    U = pca(normalized_images)[0]
    kmeans = KMeans(n_clusters=len(cnt))
    print('fitting KMEANS')

    kmeans.fit(U)

    print('stacking vectors KMEANS')


    for step, img in enumerate(normalized_images):
        if step == 0:
            vstack = img
            continue
        vstack = np.vstack((vstack, img))
    
    # print(vstack)
    prediction = kmeans.predict(vstack)
    print(prediction)
    path = os.path.join(data_root, "baseline_run_%d.pkl" % max_edge_len)
    with open(path, "wb") as f:
        eval_tup = (prediction, label_list, kmeans, vstack.shape)
        pickle.dump(eval_tup, f)

# createFeatureVectors()

for i in range(20, 400, 20):
    createFeatureVectors(i)



# def evaluate_predictions(eval_tup_path):
#     with open(eval_tup_path, "rb") as f:
#         prediction, label_list, kmeans = pickle.load(f)

#     for pred_label in prediction
