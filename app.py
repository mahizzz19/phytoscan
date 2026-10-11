from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ==============================
# Load trained model
# ==============================

model = tf.keras.models.load_model("phytoscan_best.keras")


# ==============================
# Class names
# ==============================

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]


# ==============================
# Upload folder
# ==============================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==============================
# Management recommendations
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
        "Avoid excessive moisture and prolonged leaf wetness.",
        "Provide good spacing and air circulation between plants.",
        "Remove infected plant debris from the growing area.",
        "Use locally approved fungicides when necessary and follow the product label."
    ],

    "Potato___healthy": [
        "The leaf appears healthy based on the model prediction.",
        "Continue proper watering and maintain good soil drainage.",
        "Maintain adequate spacing for good air circulation.",
        "Regularly monitor plants for early signs of disease.",
        "Remove damaged or infected leaves if they appear."
    ]
}


# ==============================
# Home page
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    recommendations = None
    image_url = None
    error = None

    # ==============================
    # When user uploads an image
    # ==============================

    if request.method == "POST":

        if "leaf_image" not in request.files:

            error = "Please select an image."

            return render_template(
                "index.html",
                error=error
            )

        file = request.files["leaf_image"]

        if file.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",
                error=error
            )

        # ==============================
        # Save uploaded image
        # ==============================

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        image_url = "/" + filepath.replace("\\", "/")


        # ==============================
        # Prepare image for CNN
        # ==============================

        img = tf.keras.utils.load_img(
            filepath,
            target_size=(128, 128)
        )

        img_array = tf.keras.utils.img_to_array(img)

        img_array = tf.expand_dims(
            img_array,
            0
        )


        # ==============================
        # Prediction
        # ==============================

        predictions = model.predict(
            img_array,
            verbose=0
        )

        score = tf.nn.softmax(
            predictions[0]
        )

        predicted_index = np.argmax(score)

        prediction = class_names[predicted_index]

        confidence = float(
            100 * np.max(score)
        )


        # ==============================
        # Get recommendations
        # ==============================

        recommendations = management.get(
            prediction,
            []
        )


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        recommendations=recommendations,
        image_url=image_url,
        error=error
    )


# ==============================
# Run Flask
# ==============================

if __name__ == "__main__":
    app.run(debug=True)