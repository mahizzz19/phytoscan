import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# 1. Dataset
# ==============================

dataset_path = "training/PlantVillage"

class_names = sorted([
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])

print("\nClasses:")
print(class_names)


# ==============================
# 2. Collect Images
# ==============================

image_paths = []
labels = []

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for class_index, class_name in enumerate(class_names):

    class_folder = os.path.join(dataset_path, class_name)

    files = [
        os.path.join(class_folder, file)
        for file in os.listdir(class_folder)
        if file.lower().endswith(valid_extensions)
    ]

    for file in files:
        image_paths.append(file)
        labels.append(class_index)


# ==============================
# 3. Same Validation Split
# ==============================

_, val_paths, _, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=123,
    stratify=labels
)

print("\nValidation image count:")

for i, class_name in enumerate(class_names):
    print(
        class_name,
        ":",
        val_labels.count(i)
    )


# ==============================
# 4. Load Images
# ==============================

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, IMG_SIZE)

    image = tf.cast(image, tf.float32)

    return image, label


validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.batch(BATCH_SIZE)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==============================
# 5. Load Best Model
# ==============================

model = tf.keras.models.load_model(
    "phytoscan_best.keras"
)

print("\nBest model loaded successfully.")


# ==============================
# 6. Evaluate
# ==============================

loss, accuracy = model.evaluate(
    validation_dataset
)

print("\nValidation Results:")
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)


# ==============================
# 7. Predictions
# ==============================

y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


# ==============================
# 8. Confusion Matrix
# ==============================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print(cm)


# ==============================
# 9. Classification Report
# ==============================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=class_names,
        zero_division=0
    )
)