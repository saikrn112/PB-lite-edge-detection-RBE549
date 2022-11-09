import tensorflow as tf
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
import argparse
import shutil
import string
from Misc_tf.MiscUtils import *
import Misc.TFUtils as tu
from Misc_tf.DataHandling import *
from Misc_tf.BatchCreationTF import *
from Misc_tf.Decorators import *
#from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
from Network.ResNet import *

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

    # If CheckPointPath doesn't exist make the path
    # if(not (os.path.isdir(CheckPointPath))):
    #    os.makedirs(CheckPointPath)
        
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
    

def ReadDirNames(ReadPath):
    """
    Inputs: 
    ReadPath is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from /content/data/TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read text files
    DirNames = open(ReadPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    return DirNames

    
def GenerateBatch(Set, indices, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(indices)-1)
        
        ImageNum += 1
    	
        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################

        I1, Label = Set[indices[RandIdx]]
        # I1 = torchvision.transforms.Normaliz
        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(I1Batch).to(device), torch.stack(LabelBatch).to(device)


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
    model_class_name = 'ResNet'
    Network = getattr('Network_tf.ResNet', 'ResNet')
    VN = Network(InputPH = InputPH, InitNeurons = Args.InitNeurons, Suffix = Args.Suffix, NumOut = Args.NumOut, UncType = Args.UncType)
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.LinearLR(Optimizer)
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath)
        # Extract only numbers from the name
        # StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        StartEpoch = 0
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
    
    model.train()
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        epochAggLoss = 0.0
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            Batch = GenerateBatch(TrainSet, train_idx, TrainLabels, ImageSize, MiniBatchSize)
            
            # Predict output with forward pass
            LossThisBatch = model.training_step(Batch)

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
            model.train()
            epochAggLoss += result['loss']
            
            # Tensorboard
            Writer.add_scalar('LossEveryIter', result["loss"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            Writer.add_scalar('Accuracy', result["acc"], Epochs*NumIterationsPerEpoch + PerEpochCounter)
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()
        
        validationBatch = GenerateBatch(TrainSet, valid_idx, TrainLabels, ImageSize, MiniBatchSize)
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

normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])

# transforms_to_apply = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),normalize])
transforms_to_apply = transforms.Compose([transforms.ToTensor(), normalize])
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
MiniBatchSize = 128
LoadCheckPoint = 0
model_type = CIFAR10Model.Model.ResNet
if model_type == CIFAR10Model.Model.Base:
    model_path = "base_model"
elif model_type == CIFAR10Model.Model.BatchNorm:
    model_path = "batch_norm"
elif model_type == CIFAR10Model.Model.ResNet:
    model_path = "resnet"
elif model_type == CIFAR10Model.Model.ResNeXt:
    model_path = "resnext"
elif model_type == CIFAR10Model.Model.DenseNet:
    model_path = "densenet"

# base_path = "../../../cnn_training/" + model_path + "/train_29_08_2022_4/"
base_path = "./"
CheckPointPath =  "./checkpoints/" + model_path + "/model.ckpt"
print(CheckPointPath)
LogsPath = base_path + "/tensor_board_logs/"
LabelsPathTrain = './TxtFiles/LabelsTrain.txt'

# if(not (os.path.isdir(LogsPath))):
#        os.makedirs(LogsPath)
# Setup all needed parameters including file reading
SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(LabelsPathTrain, CheckPointPath)

# Find Latest Checkpoint File
# if LoadCheckPoint==1:
#     LatestFile = FindLatestModel(CheckPointPath)
# else:
#     LatestFile = None

# Pretty print stats
PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, CheckPointPath)

TrainOperation(model_type, TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, CheckPointPath, TrainSet, LogsPath, train_idx, valid_idx)
