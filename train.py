import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# --------------------------------------------------
# 1. Download the flower dataset
# --------------------------------------------------

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

archive = tf.keras.utils.get_file(
    "flower_photos.tgz",
    origin=url,
    extract=True
)

# The extracted dataset contains:
# flower_photos/
#   daisy/
#   dandelion/
#   roses/
#   sunflowers/
#   tulips/

data_dir = Path(archive).parent / "flower_photos" / "flower_photos"

print("Dataset location:", data_dir)

# --------------------------------------------------
# 2. Load the images
# --------------------------------------------------

img_size = (180, 180)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names
class_names = train_ds.class_names

print("Classes:", class_names)
print("Number of classes:", len(class_names))

# --------------------------------------------------
# 3. Build the CNN
# --------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(180, 180, 3)),

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    # One output for each flower class
    layers.Dense(len(class_names), activation="softmax")
])

# --------------------------------------------------
# 4. Compile
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 5. Train
# --------------------------------------------------

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# --------------------------------------------------
# 6. Save model
# --------------------------------------------------

model.save("flower_model.keras")

print("Model saved successfully!")