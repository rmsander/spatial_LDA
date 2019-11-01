import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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