import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import  matplotlib.pyplot as plt
from collections import Counter 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


TRAIN=True

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'   
)

val_data = val_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    color_mode='grayscale'
)


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D


def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu' , input_shape=(224,224,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Conv2D(256,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2,2))

    model.add(Dropout(0.3))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

if TRAIN:
    print("Training model...")
    model = build_model()

    counter = Counter(train_data.classes)
    max_val = float(max(counter.values()))

    class_weight = {0:2.5,1:1.0}
    

    history = model.fit(
        train_data,
        epochs=20,
        validation_data=val_data,
        class_weight=class_weight
    )
    model.save("model.h5")
    print("Model save")

else:
    print("Loading saved model...")
    model = load_model("model.h5")

print("Evaluating model...")

predictions = model.predict(val_data)
predicted_classes = (predictions > 0.45).astype(int)

true_classes = val_data.classes





thresholds = np.arange(0.1, 0.9, 0.05)

for t in thresholds:
    preds = (predictions > t).astype(int)
    cm = confusion_matrix(true_classes, preds)
    print(f"\nThreshold: {t}")
    print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes))

