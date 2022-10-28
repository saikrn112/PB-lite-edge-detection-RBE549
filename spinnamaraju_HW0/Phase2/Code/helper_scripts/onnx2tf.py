#https://towardsdatascience.com/my-journey-in-converting-pytorch-to-tensorflow-lite-d244376beed
from onnx_tf.backend import prepare
import onnx

TF_PATH="./resnet_tf_model"
ONNX_PATH="./resnet.onnx"
onnx_model = onnx.load(ONNX_PATH)

 # prepare function converts an ONNX model to an internel representation
# of the computational graph called TensorflowRep and returns
# the converted representation.
tf_rep = prepare(onnx_model)

# export_graph function obtains the graph proto corresponding to the ONNX
# model associated with the backend representation and serializes
# to a protobuf file.
tf_rep.export_graph(TF_PATH)
