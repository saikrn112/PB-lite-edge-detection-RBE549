#https://towardsdatascience.com/my-journey-in-converting-pytorch-to-tensorflow-lite-d244376beed
#https://github.com/omerferhatt/torch2tflite/blob/master/torch2tflite/converter.py
#https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
import onnx
import torch
import sys
import numpy as np
sys.path.insert(1,'../Network/')

from Network import CIFAR10Model

model = CIFAR10Model(CIFAR10Model.Model.ResNet)
device = torch.device("cuda")
model.to(device)
checkpoint = torch.load('../../checkpoints/resnet/19model.ckpt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

ONNX_PATH="./resnet.onnx"

dummy_input = torch.randn(1, 3,64,64, dtype=torch.float).to(device)
dummy_output = model(dummy_input)
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


import onnxruntime
ort_session = onnxruntime.InferenceSession("resnet.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(dummy_output), ort_outs[0], rtol=1e-02, atol=1e-04)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
