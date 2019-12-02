import copy
from utils import topNError, saveErrorGraph
from models.InceptionV3 import *
from models.ResNet import *
import matplotlib.pyplot as plt
import numpy as np
# from dataset import *
from torchvision import transforms
from skimage import io
import gc
import sys
import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler

torch.backends.cudnn.benchmark = True

import numpy as np
import matplotlib.pyplot as plt
from models.ResNet import *
from models.InceptionV3 import *
from models.SegNet import *
from utils import topNError, saveErrorGraph
import copy

cnnModelPath = os.path.join('models', 'bestCNNmodel')
multiGPU = torch.cuda.device_count() > 1
cnnLr, cnnDropout = 1e-3, 0.5

# Fits image to shape (224, 224)
resnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(224, padding=False, pad_if_needed=True, fill=0,
                          padding_mode='constant'),
    transforms.ToTensor(),
    # transforms.Normalize([0.5] * 3, [0.5] * 3)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])  # for the pytorch resnet impl
])

segnet_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(224, padding=False, pad_if_needed=True, fill=0,
                          padding_mode='constant'),
    transforms.ToTensor(),
    # transforms.Normalize([0.5] * 3, [0.5] * 3)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])  # for the pytorch segnet impl
])

image_path = "/Users/yaatehr/Programs/spatial_LDA/data/ef741c8e8b81c793.jpg"


# def cnnEpoch(model, loader, device, criterion, output_period, epoch,
# optimizer=None):
#     """
#     Epoch, a subrouting of train cnn. We will only use these methods for
#         transfer learning.
#     """
#     running_loss = 0.0
#     num_batches = len(loader)
#     errors = np.zeros(2)
#     for batch_num, (inputs, labels) in enumerate(loader, 1):
#         # print(inputs)
#         # print(labels)
#         inputs = inputs.to(device)
#         labels = np.array(labels).to(device)

#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         if optimizer is not None:
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         running_loss += loss.item()

#         if batch_num % output_period == 0:
#             print('[%d:%.2f] loss: %.3f' % (
#                 epoch, batch_num*1.0/num_batches,
#                 running_loss/output_period
#             ))
#             running_loss = 0.0
#         gc.collect()
#         errors += topNError(outputs, labels, [1, 2], False)
#     return errors


# def trainCNN(modelName='resnet'):
#     """
#     for transfer learning and fine tuning cnn, should ideally not be
#     trained on
#         gogle open images dataset sicne that defeats the purpose of
#         clustering.
#     """
#     # Parameters
#     num_epochs = 25
#     output_period = 100
#     batch_size = 20
#     dataset = ImageDataset()
#     dataset_labels = dataset.get_all_labels()
#     num_classes = len(dataset_labels)

#     if modelName == 'resnet':
#         model = resnet_dropout_18(num_classes=num_classes, p=cnnDropout)
#     # elif modelName == 'inception':
#     #     model = Inception3(num_classes=num_classes, aux_logits=False)
#     else:
#         raise Exception("Please select one of \'resnet\' or \'inception\'")

#     if torch.cuda.is_available():
#         if multiGPU:
#             model = nn.DataParallel(model)
#         model = model.cuda()
#     optimizer = optim.Adam(model.parameters(), lr=cnnLr)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 'min', patience=2)
#     # setup the device for running
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     #train_loader, val_loader = dataset.frame_train_loader(batch_size,
#     model=modelName), dataset.frame_val_loader(batch_size, model=modelName)
#     train_loader, val_loader = get_loaders()
#     numTrainSamples = len(train_loader) * batch_size
#     numValSamples = len(val_loader) * batch_size

#     criterion = nn.CrossEntropyLoss().to(device)
#     epochs = np.arange(1, num_epochs+1)

#     modelStates = []

#     trainErrors, valErrors = [], []
#     bestValError, bestState = np.inf, None
#     print('Training CNN')
#     for epoch in epochs:
#         for param_group in optimizer.param_groups:
#             print('Current learning rate: ' + str(param_group['lr']))
#         model.train()
#         trainErrors.append(cnnEpoch(model, train_loader, device, criterion,
#                                     output_period, epoch,
#                                     optimizer=optimizer)/numTrainSamples)
#         gc.collect()
#         modelStates.append(
#             (model.module if multiGPU else model).state_dict().copy())
#         model.eval()
#         valErrors.append(cnnEpoch(model, val_loader, device,
#                                   criterion, output_period,
#                                   epoch)/numValSamples)
#         valError = valErrors[-1][0]
#         if valError < bestValError:
#             m = model
#             if multiGPU:
#                 m = m.module
#             if torch.cuda.is_available():
#                 m = m.cpu()
#             bestState = m.state_dict().copy()
#             torch.save(bestState, cnnModelPath + modelName)
#             model = model.to(device)
#         scheduler.step(valError)
#         gc.collect()
#         print('Epoch ' + str(epoch) + ':', 'Train error:',
#               trainErrors[-1], ', Validation error:', valErrors[-1])

#     saveErrorGraph(np.array(trainErrors), np.array(
#         valErrors), 'cnnErrors' + modelName + '.png')
#     print('Finished training CNN')
#     return bestState


# trainCNN("resnet")


def test():
    model = get_model()
    batch_size = 50
    test_img = io.imread(image_path)
    inputs = segnet_transform(test_img)
    inputs = inputs.unsqueeze(0)
    print(inputs.shape)
    # test_loader = get_single_loader(batch_size=batch_size)
    # num_batches = len(test_loader)
    # errors = np.zeros(2)
    # numSamples = len(test_loader) * batch_size
    # print(inputs.shape[1:])
    # inputs = inputs.view(inputs.shape[1:]).to(device)
    # TODO: Should these be changed?
    batchOutput = model(inputs).view(-1, 4, 128)
    batchOutput = model(inputs).view(-1, 4 * 49, 128)

    gc.collect()
    print(batchOutput.shape)


def get_model():
    model = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
    # cut off the last layer of this classifier
    new_classifier = nn.Sequential(*list(model.children())[:-2])
    model = new_classifier
    return model


def cnnEpoch(model, loader, device, criterion, output_period, epoch,
             optimizer=None):
    running_loss = 0.0
    num_batches = len(loader)
    errors = np.zeros(2)
    for batch_num, (inputs, labels) in enumerate(loader, 1):
        # print(inputs)
        # print(labels)
        inputs = inputs.to(device)
        labels = np.array(labels).to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if batch_num % output_period == 0:
            print('[%d:%.2f] loss: %.3f' % (
                epoch, batch_num * 1.0 / num_batches,
                running_loss / output_period
            ))
            running_loss = 0.0
        gc.collect()
        errors += topNError(outputs, labels, [1, 2], False)
    return errors


def trainCNN(modelName='resnet'):
    # Parameters
    num_epochs = 25
    output_period = 100
    batch_size = 20
    dataset = ImageDataset()
    dataset_labels = dataset.get_all_labels()
    num_classes = len(dataset_labels)

    if modelName == 'resnet':
        model = resnet_dropout_18(num_classes=num_classes, p=cnnDropout)
    elif modelName == 'inception':
        model = Inception3(num_classes=num_classes, aux_logits=False)
    elif modelName == 'segnet':
        # TODO: Figure out how dims need to be changed based off of NYU dataset
        model = SegNet(input_channels=3, output_channels=1, pretrained_vgg=True)
    else:
        raise Exception("Please select one of \'resnet\' or \'inception\' or "
                        "\'segnet\'")

    if torch.cuda.is_available():
        if multiGPU:
            model = nn.DataParallel(model)
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=cnnLr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           patience=2)
    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    # test()
    pass
