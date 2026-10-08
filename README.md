# Flower Classification with CNN

A small TensorFlow/Keras project that trains a convolutional neural network (CNN) to classify flower photos into five categories: daisy, dandelion, roses, sunflowers, and tulips.

## What it does

- Downloads TensorFlow's flower photos dataset when training starts.
- Trains and evaluates a CNN using an 80/20 train/validation split.
- Saves the trained model locally as `flower_model.keras`.
- Predicts the class and confidence for a flower image you provide.

## Requirements

- Python 3
- pip
- Internet access the first time you train, to download the dataset

Use a Python version supported by the TensorFlow release installed by pip. Check the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip) for current platform and Python version support.

## Setup

Clone the repository and enter its directory:

    git clone https://github.com/sahityadhiman/flower-classification.git
    cd flower-classification

Create and activate a virtual environment:

Windows PowerShell:

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1

macOS or Linux:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

## Train

Run from the repository root:

    python train.py

The script downloads the flower photos dataset to the Keras cache, trains for five epochs, evaluates on the validation split, and writes `flower_model.keras` in the current directory. The dataset is not stored in this repository.

## Predict

Train the model first, then run:

    python predict.py

When prompted, enter the path to an image. For example, use `flower.jpg` from the repository root. The script prints the predicted class and its confidence.

## Model and data

The CNN rescales images to the 0–1 range and uses three convolution/max-pooling blocks, a 128-unit dense layer, and a five-class softmax output. Images are resized to 180 × 180 pixels. The dataset contains 3,670 images in these directory labels:

- daisy
- dandelion
- roses
- sunflowers
- tulips

Dataset source: [TensorFlow flower photos archive](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz).

The archive README reports 85.39% training accuracy and 65.53% validation accuracy after five epochs. These are reported example results, not a guarantee; results can vary with the TensorFlow version and runtime.

## Repository contents

    train.py          Download data, train, evaluate, and save the model
    predict.py        Load the saved model and classify an image
    flower.jpg        Example image
    requirements.txt  Python dependencies

## Attribution and license

The supplied archive credits Mehul Maithani as the project author and links to [the original GitHub project](https://github.com/maithanmehul-bit/flower-classification). That attribution is preserved here. The archive did not include a license for the source or sample image, so this repository does not grant additional reuse rights. Please contact the original author about reuse.

## Limitations

This is an educational example. A single prediction's confidence is not a measure of overall model accuracy, and the reported validation score suggests the basic CNN may overfit. Use a separate evaluation set and improve validation performance before relying on predictions.

## Author

Original project attribution: [Mehul Maithani](https://github.com/maithanmehul-bit)

