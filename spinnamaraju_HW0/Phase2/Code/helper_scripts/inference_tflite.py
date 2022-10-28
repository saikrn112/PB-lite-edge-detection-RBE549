import tensorflow as tf
import numpy as np
import cv2

img = cv2.imread('./lite_test/5945.png')
img = cv2.resize(img,(64,64),interpolation = cv2.INTER_AREA)
img = img.swapaxes(0,2)

img2 = cv2.imread('./lite_test/404.png')
img2 = cv2.resize(img2,(64,64),interpolation = cv2.INTER_AREA)
img2 = img2.swapaxes(0,2)

img3 = cv2.imread('./lite_test/3300.png')
img3 = cv2.resize(img3,(64,64),interpolation = cv2.INTER_AREA)
img3 = img3.swapaxes(0,2)

img4 = cv2.imread('./lite_test/9008.png')
img4 = cv2.resize(img4,(64,64),interpolation = cv2.INTER_AREA)
img4 = img4.swapaxes(0,2)

TFLITE_PATH="./resnet.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print(f"{input_shape}")
dummy_input = np.array(np.random.random_sample(input_shape),dtype=np.uint8)
dummy_input[0] = img
dummy_input[1] = img2
dummy_input[2] = img3
dummy_input[3] = img4
interpreter.set_tensor(input_details[0]['index'],dummy_input)
interpreter.invoke()
y_pred = interpreter.get_tensor(output_details[0]['index'])

print(y_pred)

print(f"{y_pred.shape}")
