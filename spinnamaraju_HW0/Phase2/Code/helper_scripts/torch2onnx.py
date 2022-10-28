#https://towardsdatascience.com/my-journey-in-converting-pytorch-to-tensorflow-lite-d244376beed
#https://github.com/omerferhatt/torch2tflite/blob/master/torch2tflite/converter.py
import onnx
import torch
import sys
sys.path.insert(1,'../Network/')

from Network import CIFAR10Model

model = CIFAR10Model(CIFAR10Model.Model.ResNet)
device = torch.device("cuda")
model.to(device)
ONNX_PATH="./resnet.onnx"

dummy_input = torch.randn(16, 3,64,64, dtype=torch.float).to(device)
torch.onnx.export(
    model=model,
    args=dummy_input,
    f=ONNX_PATH,
    verbose=False,
    export_params=True,
    do_constant_folding=False,
    input_names=['input'],
    opset_version=10,
    output_names=['output']
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
