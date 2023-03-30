import tensorflow as tf
import keras
import onnx
from onnx_tf.backend import prepare
model = onnx.load('epoch_054.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)
tf_rep.export_graph('epoch_054.pb')


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('epoch_054.pb') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)