import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import numpy as np


# Load the pre-trained MobileNetV2 model
print("Loading AI model...")

model = MobileNetV2(weights="imagenet")

print("Model loaded successfully!")


# Get image path from user
image_path = input("\nEnter the image file path: ")


try:
    # Load the image
    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert image into an array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)

    # Get top 5 predictions
    results = decode_predictions(
        predictions,
        top=5
    )[0]


    print("\n" + "=" * 50)
    print("        IMAGE RECOGNITION RESULTS")
    print("=" * 50)

    for i, (_, label, probability) in enumerate(
        results,
        start=1
    ):
        print(
            f"{i}. {label} - "
            f"{probability * 100:.2f}%"
        )

    print("=" * 50)


except FileNotFoundError:
    print("\nError: Image file not found.")

except Exception as e:
    print("\nSomething went wrong:")
    print(e)
    # Project4
