import tensorflow as tf
import numpy as np
TF_PATH="./resnet_tf_model"
TFLITE_PATH="./resnet.tflite"

def representative_dataset_gen():
    for _ in range(10):
        # Get sample input data as a numpy array in a method of your choosing.
        input = np.float32(2.*(np.random.rand(1, 3, 64, 64) - 0.5))
        yield [input]

converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset_gen

tflite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

