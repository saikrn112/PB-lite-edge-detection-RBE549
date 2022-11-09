import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
#from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.MiscUtils import FindLatestModel
from Network.Network import *


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")

def SetupAll(LabelsPathTrain, CheckPointPath):
    """
    Inputs: 
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    TrainLabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Read and Setup Labels

    TrainLabels = ReadLabels(LabelsPathTrain)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100 
    
    # Image Input Shape
    ImageSize = [32, 32, 3]
    NumTrainSamples = len(TrainSet)

    # Number of classes
    NumClasses = 10

    return SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses


def ReadLabels(LabelsPathTrain):
    if(not (os.path.isfile(LabelsPathTrain))):
        print('ERROR: Train Labels do not exist in '+LabelsPathTrain)
        sys.exit()
    else:
        TrainLabels = open(LabelsPathTrain, 'r')
        TrainLabels = TrainLabels.read()
        TrainLabels = map(float, TrainLabels.split())

    return TrainLabels
    
def GenerateBatch(Set, indices, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
        Set - Variable with Subfolder paths to train files
        indices - corresponding to train or val
        TrainLabels - Labels corresponding to Train
        ImageSize - the Size of the Image
        MiniBatchSize - the size of the MiniBatch
    Outputs:
        ImagesBatch - Batch of images
        LabelBatch - Batch of one-hot encoded labels 
    """
    ImagesBatch = []
    LabelBatch = []
    
    for i in range(MiniBatchSize):
        RandIdx = random.randint(0, len(indices)-1)
        Images, Label = Set[indices[RandIdx]]
        ImagesBatch.append(Images)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(ImagesBatch).to(device), torch.stack(LabelBatch).to(device)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

def check_and_load_checkpoint(model, LatestFile, CheckPointPath):
    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + os.sep + LatestFile)

        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit())) + 1
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
    return StartEpoch
    
def TrainOperation(model_type,TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, LogsPath, train_idx, valid_idx):
    """
    Inputs: 
        TrainLabels - Labels corresponding to Train/Test
        NumTrainSamples - length(Train)
        ImageSize - Size of the image
        NumEpochs - Number of passes through the Train data
        MiniBatchSize is the size of the MiniBatch
        SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
        CheckPointPath - Path to save checkpoints/model
        DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
        LatestFile - Latest checkpointfile to continue training
        TrainSet - The training dataset
        LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Initialize the model
    model = CIFAR10Model(model_type)
    model = model.to(device)
    Optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    ## Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    StartEpoch = check_and_load_checkpoint(model, LatestFile, CheckPointPath)

    model.train()
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        epochAggLoss = 0.0
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            Batch = GenerateBatch(TrainSet, train_idx, TrainLabels, ImageSize, MiniBatchSize)
            
            # Predict output with forward pass
            LossThisBatch = model.training_step(Batch)

            model.train()
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                
                torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)

            model.eval()
            result = model.validation_step(Batch)
            epochAggLoss += result['loss']
            
            # Tensorboard
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            Writer.add_scalar('Accuracy', result["acc"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()
        
        # run on validation batch
        validationBatch = GenerateBatch(TrainSet, valid_idx, TrainLabels, ImageSize, MiniBatchSize)

        model.eval()
        validation_result = model.validation_step(validationBatch)
        Writer.add_scalar('ValidationLossEveryEpoch',validation_result['loss'],Epochs)
        Writer.add_scalar('ValidationAccuracyEveryEpoch',validation_result['acc'],Epochs)
        Writer.flush()
    
        # scheduler.step()
        model.epoch_end(Epochs, epochAggLoss ,validation_result['acc'])

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')

# Default Hyperparameters
NumEpochs = 20

transforms_to_apply = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
TrainSet = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_to_apply)

ValSet = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_to_apply)

valid_size = 0.1
num_train = len(TrainSet)
indices = list (range(num_train))
split = int(np.floor(valid_size * num_train))

np.random.seed(42)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]


DivTrain = 1.0
MiniBatchSize = 2048
LoadCheckPoint = 0
model_type = CIFAR10Model.Model.ResNet

base_path = "../"
CheckPointPath =  "../checkpoints/resnet/"
print(CheckPointPath)
LogsPath = base_path + "/tensor_board_logs/"
LabelsPathTrain = './TxtFiles/LabelsTrain.txt'

SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(LabelsPathTrain, CheckPointPath)

# Find Latest Checkpoint File
if LoadCheckPoint==1:
    LatestFile = FindLatestModel(CheckPointPath)
else:
    LatestFile = None

# Pretty print stats
PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, CheckPointPath)

TrainOperation(model_type, TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, LogsPath, train_idx, valid_idx)

