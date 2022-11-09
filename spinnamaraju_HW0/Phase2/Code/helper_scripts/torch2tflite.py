#https://towardsdatascience.com/my-journey-in-converting-pytorch-to-tensorflow-lite-d244376beed
#https://github.com/omerferhatt/torch2tflite/blob/master/torch2tflite/converter.py
#https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(1,'../Network/')

import numpy as np
import torch
import onnxruntime
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
from Network import CIFAR10Model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def representative_dataset_gen():
    for _ in range(10):
        # Get sample input data as a numpy array in a method of your choosing.
        input = np.float32(2.*(np.random.rand(1, 3, 64, 64) - 0.5))
        yield [input]


ONNX_PATH="./resnet.onnx"
TF_PATH="./resnet_tf_model"
TFLITE_PATH="./resnet.tflite"
CHECKPOINT_PATH='../../checkpoints/resnet/19model.ckpt'

model = CIFAR10Model(CIFAR10Model.Model.ResNet)
device = torch.device("cuda")
checkpoint = torch.load(CHECKPOINT_PATH)

model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

###### CONVERTING TORCH to ONNX
dummy_input = torch.randn(1, 3,64,64, dtype=torch.float).to(device)
dummy_output = model(dummy_input)
torch.onnx.export(model=model, args=dummy_input, f=ONNX_PATH, verbose=False,
                  export_params=True,
                  do_constant_folding=False,
                  input_names=['input'],
                  opset_version=10,
                  output_names=['output']
                 )

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(ONNX_PATH, providers=['TensorrtExecutionProvider', 
                                                                'CUDAExecutionProvider', 
                                                                'CPUExecutionProvider'])
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(dummy_output), ort_outs[0], rtol=1e-02, atol=1e-04)

print("Exported model has been tested with ONNXRuntime, and the result looks good!\n\n\n")


###### CONVERTING ONNX to TORCH
#https://towardsdatascience.com/my-journey-in-converting-pytorch-to-tensorflow-lite-d244376beed
onnx_model = onnx.load(ONNX_PATH)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(TF_PATH)

###### CONVERTING TF to TFLITE
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"exported to TFLite")
