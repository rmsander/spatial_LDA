import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
from skimage.color import rgb2gray
import copy
from collections import Counter
import sklearn as skl
from crop_images import *
from utils import *
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import numpy as np

data_root = os.path.join(os.path.dirname(__file__), '../data')
# train_root = os.path.join(data_root, 'train')
# val_root = os.path.join(data_root, 'val')
# test_root = os.path.join(data_root, 'test')
# hierarchy_json_path = os.path.join(data_root,
# 'bbox_labels_600_hierarchy.json')

train_root = "/home/yaatehr/programs/spatial_LDA/data/cropped_test_0/m"
test_root = "/home/yaatehr/programs/spatial_LDA/data/cropped_test_0/m"
hierarchy_json_path = "/home/yaatehr/programs/spatial_LDA/data" \
                      "/bbox_labels_600_hierarchy.json"
path_to_csv = "/home/yaatehr/programs/datasets/google_open_image/train" \
              "-annotations-bbox.csv"
path_to_classname_map_csv = os.path.join(data_root, 'class-descriptions.csv')



def create_classname_map(path_to_csv):
    output = {}
    with open(path_to_csv, 'r') as file:
        line = file.readline()
        while line:
            vals = line.split(",")
            output[vals[0].split("/")[-1]] = vals[1].strip()
            line = file.readline()
    return output


try:
    classname_map = create_classname_map(path_to_classname_map_csv)
    max_hierarchy_level = 3
    granularity_map = make_inverted_labelmap(max_hierarchy_level,
                                            path_to_hierarchy=hierarchy_json_path)
except Exception as e:
    classname_map = None
    granularity_map = None
    # print("could not initialize classname map or granularity map for google open images")


resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(224, pad_if_needed=True, fill=0, padding='constant'),
    transforms.ToTensor(),
    # transforms.Normalize([0.5] * 3, [0.5] * 3)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])  # for the pytorch resnet impl
])
#  Note: these constants are the normalization constants for normalize
segnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(224, pad_if_needed=True, fill=0, padding='constant'),
    transforms.ToTensor(),
    # transforms.Normalize([0.5] * 3, [0.5] * 3)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])  # for the pytorch segnet impl
])


class ADE20K(Dataset):

    def __init__(self, root=train_root, transform=resnet_transform, grayscale=False, numLabelsLoaded=0, labelSubset=None, useStringLabels=True, randomSeed=33, normalizeWeights=False):
        """
        Args:
            root_dir (string): Directory with all the images organized into
            folders by class label (hash).
            transform (callable, optional): Optional transform to be applied
                on a sample.
            grayscale: convert to greyscale (load with skimage imread)
            numLabelsLoaded: pick a subset of labels (ran tune with randomSeed)
            labelSubset: manually pick labels, overrides numLabelsLoaded
            useStringLabels: use string over one hot
            normalizeWeights: if label subsets in effct, will make sure there are equal numbers of each class


        """
        super(Dataset, self).__init__()

        self.root = root
        self.transform = transform
        self.grayscale = grayscale
        self.useStringLabels = useStringLabels
        self.image_paths = []
        self.class_indices = {}
        self.image_classes = []
        self.counter = Counter()
        index = 0
        for (dirpath, dirnames, filenames) in os.walk(self.root):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    self.image_paths.append(os.path.join(dirpath, filename))
                    label = os.path.basename(dirpath).split("/")[-1]
                    self.image_classes.append(label)
                    # print(label)
                    self.counter[label]+=1
                    if label in self.class_indices.keys():
                        self.class_indices[label].append(index)
                    else:
                        self.class_indices[label] = [index]
                    index +=1

    
        self.class_set = list(self.class_indices.keys())

        #select subset of classes
        if labelSubset is not None or numLabelsLoaded > 0: 
            #for manually selected strings
            if labelSubset is not None:              
                indToRemove = copy.copy(self.class_indices)
                existing_classes = set(self.image_classes)
                for label in labelSubset:
                    if label not in existing_classes:
                        raise Exception("Invalid class name in labelSubset: " + label)
                    del indToRemove[label]
            #for random subset fo classes
            elif(numLabelsLoaded > 0):
                np.random.seed(randomSeed)
                indices = list(range(len(self.class_set)))
                np.random.shuffle(indices)
                labelSubset = [self.class_set[j] for i, j in enumerate(indices) if i < numLabelsLoaded]
                print(len(labelSubset))
                print(labelSubset)
                indToRemove = copy.copy(self.class_indices)
                for label in labelSubset:
                    del indToRemove[label]
            indices = []
            for l in [indToRemove[key] for key in indToRemove.keys()]:
                indices.extend(l)
            indices = np.array(indices)
            # print(indices.shape)
            self.image_paths = np.delete(np.array(self.image_paths), indices).tolist()
            self.image_classes = np.delete(np.array(self.image_classes), indices).tolist()
            index = 0
            min_label_samples = min([len(self.class_indices[i]) for i in labelSubset])
            self.class_indices = {}
            self.counter = Counter()
            new_impaths = []
            new_labels = []
            for i, path in enumerate(self.image_paths):
                label = os.path.basename(os.path.dirname(path)).split("/")[-1]

                if(normalizeWeights):
                    if self.counter[label] == min_label_samples:
                        continue
                    new_impaths.append(path)
                    new_labels.append(self.image_classes[i])
                self.counter[label] +=1
                if label in self.class_indices.keys():
                    self.class_indices[label].append(index)
                else:
                    self.class_indices[label] = [index]
                index +=1 
            if(normalizeWeights):
                self.image_paths = new_impaths
                self.clas_labels = new_labels
            
        print("Loaded ADE20K ewith follwoing distribution: ", self.counter)

        self.onehot_labelmap = self.init_one_hot_map(list(self.class_indices.keys()))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        impath = self.image_paths[idx]
        image = io.imread(impath, as_gray=self.grayscale)
        if self.transform:
            image = self.transform(image)
        image_class_hash = os.path.basename(os.path.dirname(impath)).split("/")[
            -1]

        label = self.image_classes[idx]
        if not self.useStringLabels:
            label = self.onehot_labelmap[label]

        return image, label

    def get_all_label_strings(self, use_text=True):
        output = set(self.image_classes)
        return output
    def get_onehot_label(self, label):
        return self.onehot_labelmap[label]

    def init_one_hot_map(self, data):
        label_encoder = skl.preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data)
        # binary encode
        onehot_encoder = skl.preprocessing.OneHotEncoder(sparse=False, categories='auto')
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return dict(zip(data, onehot_encoded))


class ImageDataset(Dataset):

    def __init__(self, root=train_root, transform=resnet_transform, grayscale=False):
        """
        Args:
            root_dir (string): Directory with all the images organized into
            folders by class label (hash).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(Dataset, self).__init__()

        self.root = root
        self.transform = transform
        self.grayscale = grayscale

        self.image_paths = []
        for (dirpath, dirnames, filenames) in os.walk(self.root):
            self.image_paths.extend([os.path.join(dirpath, filename)
                                     for filename in filenames if
                                     filename.endswith('.jpg')])
        self.image_class_hashes = [
            os.path.basename(os.path.dirname(impath)).split("/")[
                -1] for impath in self.image_paths]
        self.effective_hashes = [granularity_map[imhash]
                                 for imhash in self.image_class_hashes]
        self.effective_labels = [classname_map[imhash]
                                 for imhash in self.effective_hashes]
        self.onehot_labelmap = self.init_one_hot_map(classname_map.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        impath = self.image_paths[idx]
        image = io.imread(impath, as_gray=self.grayscale)
        if self.transform:
            image = self.transform(image)
        # TODO: Splitting might mess this up
        image_class_hash = os.path.basename(os.path.dirname(impath)).split("/")[
            -1]

        label = self.effective_labels[idx]
        return image, self.onehot_labelmap[label]

    def get_all_label_strings(self, use_text=True):
        output = set(self.effective_labels)
        return output

    def init_one_hot_map(self, data):
        label_encoder = skl.preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data)
        # binary encode
        onehot_encoder = skl.preprocessing.OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return dict(zip(data, onehot_encoded))


def getImageLabel(classname, use_text=True):
    image_class = granularity_map[classname]
    if (use_text):
        return classname_map[image_class]
    return image_class


def get_loaders(dataset=None, batch_size=50, validation_split=.2,
                random_seed=54, shuffle_dataset=True):
    """
        returns a train and validation loader from a single dataset.
    """
    if not dataset:
        dataset = ImageDataset(train_root)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             sampler=valid_sampler)

    return train_loader, val_loader


def get_single_loader(dataset=None, batch_size=50, shuffle_dataset=False, random_seed=54):
    """
        returns a single data loader, should be used for test dataset
    """
    if not dataset:
        dataset = ImageDataset(test_root)
    
    # if getattr(dataset, 'counter', None) != None:
    #     sampler = WeightedRandomSampler(list(np.array([i for i in dataset.counter.values()])/sum(dataset.counter.values())), 100)
    # else:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
