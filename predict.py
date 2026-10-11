import tensorflow as tf
import numpy as np

# ==============================
# 1. Load the trained model
# ==============================

model = tf.keras.models.load_model("phytoscan_best.keras")


# ==============================
# 2. Class names
# ==============================

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]


# ==============================
# 3. Management Recommendations
# ==============================

management = {

    "Potato___Early_blight": [
        "Remove and properly dispose of severely infected leaves.",
        "Maintain adequate spacing between plants for good air circulation.",
        "Avoid prolonged leaf wetness and unnecessary overhead irrigation.",
        "Remove infected plant debris from the growing area.",
        "Use locally approved fungicides when necessary and follow the product label."
    ],

    "Potato___Late_blight": [
        "Remove and properly dispose of infected plant material.",
        "Avoid overhead irrigation and prolonged leaf wetness.",
        "Provide good air circulation by maintaining proper plant spacing.",
        "Monitor nearby plants regularly because late blight can spread quickly.",
        "Use locally approved fungicides when necessary and follow agricultural guidance."
    ],

    "Potato___healthy": [
        "No visible disease was detected by the model.",
        "Continue regularly monitoring the potato plants.",
        "Maintain proper irrigation and avoid excessive moisture on leaves.",
        "Maintain good spacing and air circulation between plants.",
        "Remove diseased plant material promptly if symptoms appear."
    ]
}


# ==============================
# 4. Test image
# ==============================

image_path = "test_images/potato_test.JPG"


# ==============================
# 5. Load and preprocess image
# ==============================

img = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128)
)

img_array = tf.keras.utils.img_to_array(img)

img_array = tf.expand_dims(img_array, 0)


# ==============================
# 6. Make prediction
# ==============================

predictions = model.predict(
    img_array,
    verbose=0
)

score = predictions[0]

predicted_index = np.argmax(score)

predicted_class = class_names[predicted_index]

confidence = 100 * score[predicted_index]


# ==============================
# 7. Display result
# ==============================

print("\n========================================")
print("             PHYTOSCAN")
print("========================================")

print("\nPrediction:", predicted_class)
print("Confidence:", round(confidence, 2), "%")


# ==============================
# 8. Display management advice
# ==============================

print("\nManagement Recommendations:")

for recommendation in management[predicted_class]:
    print("•", recommendation)

print("\n========================================")