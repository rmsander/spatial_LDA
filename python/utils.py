import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from os import path
import json

def topNError(output, labels, ns, percent=True):
    sortedOutputs = output.topk(k = max(ns), dim=1, sorted=True)[1]
    topNs = torch.cat((sortedOutputs, labels.view(labels.shape + (1,))), dim=1)
    results = [[row[-1] in row[:n] for row in torch.unbind(topNs, dim=0)] for n in ns]
    errors = [len(res) - np.sum(res) for res in results]
    return np.array([error/len(labels) for error in errors] if percent else errors)

def confusionMatrix(outputs, labels):
    return confusion_matrix(labels, torch.argmax(outputs, dim=1))

def saveErrorGraph(trainErrors, valErrors, outfile):
    trainClassificationErrors, trainTop2Errors = trainErrors[:,0], trainErrors[:,1]
    valClassificationErrors, valTop2Errors = valErrors[:,0], valErrors[:,1]
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training and Validation Errors')
    epochs = np.arange(1, trainErrors.shape[0] + 1)
    plt.plot(epochs, trainClassificationErrors, label="Train Classification Error")
    plt.plot(epochs, trainTop2Errors, label="Train Top 2 Error")
    plt.plot(epochs, valClassificationErrors, label="Validation Classification Error")
    plt.plot(epochs, valTop2Errors, label="Validation Top 2 Error")
    plt.legend(loc='best')
    plt.savefig(outfile)

LABEL_HIERARCHY_PATH = "bbox_labels_600_hierarchy.json"

def build_tree_to_depth_n(root, n):
    queue = [root]
    output = dict()
    label_map = dict()
    num_nodes_in_layer = 1
    depth = 0

    while len(queue) > 0:
        node = queue.pop(0)
        num_nodes_in_layer -= 1
        if(depth < n):
            label_map[node["LabelName"]] = node["LabelName"]
        else:
            label_map[node["LabelName"]] = get_all_sublabels(node)
        try:
            childList = node["Subcategory"]
            for child in childList:
                queue.append(child)
        except:
            pass
        if(num_nodes_in_layer == 0) :
            num_nodes_in_layer = len(queue)
            depth += 1
        
        if(depth > n):
            break

    #invert the dictionary (map from lower level labels to higher)
    for key, value in label_map.items():
        if isinstance(value, str):
            output[key.split("/")[-1]] = value.split("/")[-1]
        else:
            for label in value:
                if label:
                    output[label.split("/")[-1]] = key.split("/")[-1]
        
    return output

def get_all_sublabels(node):
    sublabels = []
    queue = [node]
    while len(queue) > 0:
        n = queue.pop()
        sublabels.append(n["LabelName"])
        try:
            childList = n["Subcategory"]
            for child in childList:
                queue.append(child)
        except:
            pass
    return sublabels


def make_inverted_labelmap(depth, path_to_hierarchy=LABEL_HIERARCHY_PATH):
    if path.exists(path_to_hierarchy):
        hierarchy_json_tree = json.load(open(path_to_hierarchy, 'r'))
        # hierarchy_json_tree = jsontree.load(open(LABEL_HIERARCHY_PATH, 'r'))
        node_map = build_tree_to_depth_n(hierarchy_json_tree, depth)
        return node_map
    else:
        raise Exception("counld not find label json hierarchy")

# make_inverted_labelmap(2)