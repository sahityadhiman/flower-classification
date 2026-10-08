import tensorflow as tf
import numpy as np

# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

model = tf.keras.models.load_model("flower_model.keras")

# The classes used during training
class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]

# --------------------------------------------------
# 2. Ask for an image
# --------------------------------------------------

image_path = input("Enter the path of the flower image: ")

# --------------------------------------------------
# 3. Load and prepare the image
# --------------------------------------------------

img = tf.keras.utils.load_img(
    image_path,
    target_size=(180, 180)
)

img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = tf.expand_dims(img_array, 0)

# --------------------------------------------------
# 4. Make prediction
# --------------------------------------------------

predictions = model.predict(img_array)

# Get the class with the highest probability
predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100

# --------------------------------------------------
# 5. Display result
# --------------------------------------------------

print("\n🌸 Prediction Result")
print("-------------------------")
print("Predicted flower:", predicted_class)
print(f"Confidence: {confidence:.2f}%")