import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================================
# LOAD MODEL
# =========================================

model = load_model("mobilenet_pneumonia.keras")

# =========================================
# IMAGE PATH
# =========================================

img_path = 'dataset/test/PNEUMONIA/person1_virus_6.jpeg'

IMG_SIZE = 224

# =========================================
# LOAD AND PREPROCESS IMAGE
# =========================================

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = img_array / 255.0

# =========================================
# PREDICTION
# =========================================

prediction = model.predict(img_array)[0][0]

print("Prediction Probability:", prediction)

if prediction > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")

# =========================================
# LAST CONVOLUTION LAYER
# =========================================

last_conv_layer_name = "Conv_1"

last_conv_layer = model.get_layer(last_conv_layer_name)

# =========================================
# CREATE GRAD MODEL
# =========================================

grad_model = tf.keras.models.Model(
    [model.inputs],
    [last_conv_layer.output, model.output]
)

# =========================================
# COMPUTE GRADIENTS
# =========================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img_array)

    loss = predictions[:, 0]

grads = tape.gradient(loss, conv_outputs)

# =========================================
# GLOBAL AVERAGE POOLING
# =========================================

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

# =========================================
# WEIGHT FEATURE MAPS
# =========================================

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

# =========================================
# NORMALIZE HEATMAP
# =========================================

heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)

# =========================================
# DISPLAY HEATMAP
# =========================================

plt.matshow(heatmap)

plt.title("Grad-CAM Heatmap")

plt.show()

# =========================================
# OVERLAY HEATMAP ON IMAGE
# =========================================

img = cv2.imread(img_path)

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img

# =========================================
# SHOW FINAL RESULT
# =========================================

plt.figure(figsize=(8,8))

plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))

plt.title("Grad-CAM Overlay")

plt.axis('off')

plt.show()