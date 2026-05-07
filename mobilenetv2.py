import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================================
# CONTROL SWITCH
# =========================================

TRAIN = True

# =========================================
# IMAGE SIZE + BATCH
# =========================================

IMG_SIZE = 224
BATCH_SIZE = 32

# =========================================
# DATA GENERATORS
# =========================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================================
# LOAD DATA
# =========================================

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb'
)

val_data = val_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

# =========================================
# BUILD MODEL FUNCTION
# =========================================

def build_model():

    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze pretrained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Custom classification head
    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)

    x = Dropout(0.3)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# =========================================
# TRAIN OR LOAD MODEL
# =========================================

if TRAIN:

    print("Building model...")

    model = build_model()

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    print("Training model...")

    history = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        callbacks=[early_stop]
    )

    # Save model
    model.save("mobilenet_pneumonia.keras")

    print("Model saved!")

else:

    print("Loading saved model...")

    model = load_model("mobilenet_pneumonia.keras")

# =========================================
# EVALUATION
# =========================================

print("Evaluating model...")

predictions = model.predict(val_data)

# Convert probabilities to classes
predicted_classes = (predictions > 0.5).astype(int)

true_classes = val_data.classes

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(true_classes, predicted_classes)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['NORMAL', 'PNEUMONIA'],
    yticklabels=['NORMAL', 'PNEUMONIA']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))

# =========================================
# SINGLE IMAGE PREDICTION
# =========================================

from tensorflow.keras.preprocessing import image

img_path = 'dataset/test/PNEUMONIA/person1_virus_6.jpeg'

img = image.load_img(
    img_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print("\nSingle Image Prediction Probability:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")