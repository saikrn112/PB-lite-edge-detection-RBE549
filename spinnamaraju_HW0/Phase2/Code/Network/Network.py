from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch 
from enum import Enum

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.normalize = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample:
            residual = self.downsample(x)
        
        out += residual

        out = self.relu(out)
        return out

class ResNeXtBlock(nn.Module):
    def __init__ (self, in_channels, bottlenect_width,out_channels, stride, cardinality):
        super().__init__()
        actual_width = bottlenect_width*cardinality
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels,actual_width,kernel_size=1,stride=1),
                nn.BatchNorm2d(actual_width),
                nn.ReLU()
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(actual_width,actual_width,kernel_size=3,stride=stride,padding=1,groups=cardinality),
                nn.BatchNorm2d(actual_width),
                nn.ReLU()
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(actual_width,out_channels,kernel_size=1),
                nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or out_channels/in_channels != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        nn.relu_out = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        residual = self.shortcut(x)
        out += residual
        return out

class DenseNetBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_width, growth_rate, drop_rate):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,bottleneck_width*growth_rate,kernel_size=1,stride=1,bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(bottleneck_width*growth_rate),
            nn.ReLU(),
            nn.Conv2d(bottleneck_width*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False)
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        else:
            x = x
        out = torch.cat(x,1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return out

class DenseNetLayer(nn.Module):
    def __init__(self,in_channels, bottleneck_width, growth_rate, drop_rate, n_layers):
        super().__init__()
        denseNetBlocks = []
        for i in range(n_layers):
            denseNetBlocks.append(DenseNetBlock(in_channels + growth_rate*i, bottleneck_width,growth_rate,drop_rate).to(torch.device("cuda")))
        self.denseNetBlocks = denseNetBlocks

    def forward(self,x):
        x = [x]
        for denseNetBlock in self.denseNetBlocks:
            out = denseNetBlock(x)
            x.append(out)
        x1 = torch.cat(x,1)
        return x1


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out,labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, loss, acc):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, loss,acc))


class CIFAR10Model(ImageClassificationBase):
    class Model(Enum):
        Base = 1
        BatchNorm = 2 
        ResNet = 3
        ResNeXt = 4
        DenseNet = 5
    
    def baseModel(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Output size (32 x 32 x 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), # Output size (32 x 32 x 32)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # Output (16 x 16 x 32)
            nn.Conv2d(32, 64, 3, padding=1), # output(16 x 16 x 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), # output (16 x 16 x 128)
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output (8 x 8 x 128)
            nn.Flatten(),
            nn.Linear(8*8*128,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

    def batchNormModel(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # Output size (64 x 64 x 16)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), # Output size (64 x 64 x 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # Output (32 x 32 x 32)
            nn.Conv2d(32, 64, 3, padding=1), # output(32 x 32 x 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), # output (32 x 32 x 128)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output (16 x 16 x 128)
            
            nn.Conv2d(128, 256, 3, padding=1), # output(16 x 16 x 256)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1), # output (16 x 16 x 512)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # output (8 x 8 x 512)

            nn.Flatten(),
            nn.Linear(8*8*512,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,10)
        )
    
    ## RESNET ###
    def make_layer_resnet(self, in_channels,out_channels,n_blocks, stride, downsample):
        resnet_layers = []
        downsample_layer = None
        
        if downsample:
            downsample_layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        resnet_layers.append(ResNetBlock(in_channels,out_channels,stride,downsample_layer))
        for i in range(1,n_blocks):
            resnet_layers.append(ResNetBlock(out_channels,out_channels,stride=1)) # subsequent blocks shouldnt decrease image size
    
        return nn.Sequential(*resnet_layers)
    
    def initResNet(self,num_classes = 10):
        self.layer0 = self.make_layer_resnet(3, 16, 3, 2, downsample = True)
        self.layer1 = self.make_layer_resnet(16, 32, 4, 2, downsample = True)
        self.layer2 = self.make_layer_resnet(32, 64, 3, 2, downsample = True)
        self.layer3 = self.make_layer_resnet(64, 128, 3, 2, downsample = True)
        self.flatten = nn.Flatten()
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.7)
    
    def forwardResNet(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    ## RESNEXT ###
    def _make_resnext_layers(self,in_channels,block_channels,out_channels,stride,cardinality,n_layers):
        resnext_layers = []
        resnext_layers.append(ResNeXtBlock(in_channels,block_channels,out_channels,stride,cardinality))
        for i in range(1,n_layers):
            resnext_layers.append(ResNeXtBlock(out_channels,block_channels,out_channels,stride,cardinality))
        return nn.Sequential(*resnext_layers)

    def initResNeXt(self):
        self.layer0 = self._make_resnext_layers(3,4,16,2,8,3)
        self.layer1 = self._make_resnext_layers(16,4,32,2,8,4)
        self.layer2 = self._make_resnext_layers(32,4,64,2,8,3)
        self.layer3 = self._make_resnext_layers(64,4,128,2,8,3)
        self.avgpool = nn.AvgPool2d(4,stride =1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128,10)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.7)

    def forwardResNeXt(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    ## DENSENET ###
    def initDenseNet(self):
        self.conv1 = nn.Sequential(
                    nn.Conv2d(3,16,kernel_size=3,padding=1,stride=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU()
                    # nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        growth_rate = 16
        n_layers_list = [2,3]
        dropout_list = [0.2,0.7]
        layers = []
        in_channels = 16
        bn_width = 4
        actual_in_channels = in_channels
        for i,n_layers in enumerate(n_layers_list):
            layers.append(
                DenseNetLayer(actual_in_channels,bn_width,growth_rate,dropout_list[i],n_layers)
            )
            actual_in_channels = actual_in_channels + n_layers*growth_rate
            if i != len(n_layers_list) -1 :
                layers.append(nn.BatchNorm2d(actual_in_channels))
                layers.append(nn.ReLU())
                layers.append(nn.Conv2d(actual_in_channels,actual_in_channels // 2,kernel_size=1,stride=1))
                layers.append(nn.AvgPool2d(kernel_size=2,stride=2))
                actual_in_channels = actual_in_channels // 2
        layers.append(nn.BatchNorm2d(actual_in_channels))
        layers.append(nn.ReLU())
        self.final_conv = nn.Conv2d(actual_in_channels,actual_in_channels,kernel_size=2,stride=2)
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(actual_in_channels,10)
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    
    def forwardDenseNet(self,x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
    def __init__(self, model = Model.Base):
        super().__init__()
        self.model = model
        """
        Inputs: 
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        #############################
        # Fill your network initialization of choice here!
        #############################
        if model == self.Model.Base:
            self.network = self.baseModel()
        elif model == self.Model.BatchNorm:
            self.network = self.batchNormModel()
        elif model == self.Model.ResNet:
            self.initResNet()
        elif model == self.Model.ResNeXt:
            self.initResNeXt()
        elif model == self.Model.DenseNet:
            self.initDenseNet()
      
    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        if self.model == self.Model.Base:
            out = self.network(xb)
        elif self.model == self.Model.BatchNorm:
            out = self.network(xb)
        elif self.model == self.Model.ResNet:
            out = self.forwardResNet(xb)
        elif self.model == self.Model.ResNeXt:
            out = self.forwardResNeXt(xb)
        elif self.model == self.Model.DenseNet:
            out = self.forwardDenseNet(xb)
        return out


# class ResNet(ImageClassificationBase):
    # def __init__(self, block, layers, num_classes = 10):
    #     super().__init__()
    #     self.inplanes = 64
    #     self.conv1 = nn.Sequential(
    #                     nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
    #                     nn.BatchNorm2d(64),
    #                     nn.ReLU())
    #     self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    #     self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
    #     self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
    #     # self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
    #     # self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
    #     self.avgpool = nn.AvgPool2d(7, stride=1)
    #     self.fc = nn.Linear(61952, num_classes)
    
    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
    #             nn.BatchNorm2d(planes),
    #         )
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))

    #     return nn.Sequential(*layers)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.maxpool(x)
    #     x = self.layer0(x)
    #     x = self.layer1(x)
    #     # x = self.layer2(x)
    #     # x = self.layer3(x)

    #     x = self.avgpool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)

    #     return x

    # def __init__(self, num_classes = 10):
    #     # simplified version of resnet to work with 32x32 images
    #     super().__init__()
    #     self.layer0 = self._make_layer2(3, 16, 3, 2, downsample = True)
    #     self.layer1 = self._make_layer2(16, 32, 4, 2, downsample = True)
    #     self.layer2 = self._make_layer2(32, 64, 3, 2, downsample = True)
    #     self.flatten = nn.Flatten()
    #     self.avgpool = nn.AvgPool2d(4, stride=1)
    #     self.fc = nn.Linear(64, num_classes)
    #     self.dropout = nn.Dropout(0.5)
    
    # def _make_layer2(self, in_channels,out_channels,n_blocks, stride, downsample):
    #     resnet_layers = []
    #     downsample_layer = None
        
    #     if downsample:
    #         downsample_layer = nn.Sequential(
    #             nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
    #             nn.BatchNorm2d(out_channels)
    #         )
        
    #     resnet_layers.append(ResNetBlock(in_channels,out_channels,stride,downsample_layer))
    #     for i in range(1,n_blocks):
    #         resnet_layers.append(ResNetBlock(out_channels,out_channels,stride=1)) # subsequent blocks shouldnt decrease image size
        
    #     return nn.Sequential(*resnet_layers)
    
    
    # def forward(self, x):
    #     x = self.layer0(x)
    #     x = self.dropout(x)
    #     x = self.layer1(x)
    #     x = self.dropout(x)
    #     x = self.layer2(x)
    #     x = self.dropout(x)
    #     x = self.avgpool(x)
    #     x = x.view(x.size(0),-1)
    #     x = self.fc(x)
    #     return x