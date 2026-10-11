import os
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# ==============================
# 1. Dataset
# ==============================

dataset_path = "training/PlantVillage"

print("Dataset exists:", os.path.exists(dataset_path))

class_names = sorted([
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder))
])

print("\nClasses:")
print(class_names)

# Get all image paths and labels
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

    print(f"{class_name} : {len(files)} images")

    for file in files:
        image_paths.append(file)
        labels.append(class_index)


print("\nTotal images:", len(image_paths))


# ==============================
# 2. Stratified Train/Validation Split
# ==============================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    random_state=123,
    stratify=labels
)

print("\nTraining images:", len(train_paths))
print("Validation images:", len(val_paths))


# ==============================
# 3. Show validation class counts
# ==============================

print("\nValidation image count:")

for i, class_name in enumerate(class_names):
    count = val_labels.count(i)
    print(f"{class_name} : {count}")


# ==============================
# 4. Image loading function
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


# ==============================
# 5. Create TensorFlow datasets
# ==============================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.shuffle(
    buffer_size=len(train_paths),
    seed=123
)

train_dataset = train_dataset.batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)


# ==============================
# 6. Data Augmentation
# ==============================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

print("\nData augmentation ready.")


# ==============================
# 7. CNN Model
# ==============================

model = tf.keras.Sequential([

    tf.keras.Input(shape=(128, 128, 3)),

    data_augmentation,

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])

model.summary()


# ==============================
# 8. Compile Model
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel compiled successfully.")


# ==============================
# 9. Class Weights
# ==============================

class_weights = {
    0: 1.0,   # Early Blight
    1: 1.0,   # Healthy
    2: 1.0    # Late Blight
}

print("Class weights:", class_weights)


# ==============================
# 10. Save Best Model
# ==============================

checkpoint = ModelCheckpoint(
    "phytoscan_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)


# ==============================
# 11. Early Stopping
# ==============================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)


# ==============================
# 12. Train Model
# ==============================

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)

print("\nTraining completed.")
print("Best model saved as: phytoscan_best.keras")