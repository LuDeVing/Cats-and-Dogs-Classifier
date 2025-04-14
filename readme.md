# Cat vs. Dog Image Classifier 🐱🐶

![Coding Cat GIF](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)

Welcome to the Cat vs. Dog Image Classifier project! This script uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to distinguish between images of cats and dogs.

It handles image preprocessing, data preparation, model training, evaluation, and saving/loading the trained model.

## ✨ Features

* **Image Preprocessing:** Reads images, converts them to grayscale, and resizes them to a target size (128x128).
* **Efficient Data Handling:** Creates and uses an `.npz` file (`data/data.npz`) to cache preprocessed image data, speeding up subsequent runs.
* **CNN Model:** Implements a standard CNN architecture with Convolutional, MaxPooling, Flatten, Dense, and Dropout layers.
* **Training & Evaluation:** Trains the model on your image data and evaluates its accuracy on a held-out test set.
* **Model Persistence:** Saves the trained model to `data/model.h5` and loads it automatically if it already exists, avoiding retraining.
* **Basic Prediction:** Includes commented-out code to predict the class of a single image.

## 📁 Project Structure

Make sure your project directory is set up like this:

```
your-project-root/
├── data/
│   ├── images/
│   │   ├── cats/         # Folder for cat images
│   │   │   ├── cat1.jpg
│   │   │   ├── cat2.png
│   │   │   └── ...
│   │   ├── dogs/         # Folder for dog images
│   │   │   ├── dog1.jpeg
│   │   │   ├── dog2.jpg
│   │   │   └── ...
│   │   └── test_images/  # Optional: For single image predictions
│   │       └── img.png
│   ├── data.npz          # Generated automatically (dataset cache)
│   └── model.h5          # Generated automatically (trained model)
└── image_classifier.py
```

**Important:**
* The names of the folders inside `data/images/` (e.g., `cats`, `dogs`) are used as the class labels. The script currently assumes **two** classes based on the `Dense(2, ...)` output layer.
* Place your training images into the respective class folders.

## ⚙️ Requirements

You'll need Python 3 and the following libraries:

* TensorFlow / Keras
* NumPy
* Pillow (PIL Fork)
* tqdm (for progress bars)

You can install them using pip:

```bash
pip install -r requirements.txt
```
## 🚀 Usage

1.  **Prepare Data:**
    * Create the directory structure shown above (`data/images/cats`, `data/images/dogs`).
    * Populate these folders with your cat and dog images (various formats like JPG, PNG should work). The more images, the better the potential performance!

2.  **Run the Script:**
    * Navigate to `your-project-root` in your terminal.
    * Execute the Python script:
        ```bash
        python image_classifier.py
        ```
    * **First Run:** The script will:
        * Read images from `data/images/`.
        * Preprocess them (resize, grayscale).
        * Save the processed data into `data/data.npz`.
        * Split data into training (80%) and testing (20%).
        * Build and compile the CNN model.
        * Train the model for 25 epochs (you'll see progress bars).
        * Save the trained model to `data/model.h5`.
        * Evaluate the model on the test set and print the accuracy.
    * **Subsequent Runs:** The script will:
        * Detect `data/data.npz` and load data directly from it (faster).
        * Detect `data/model.h5` and load the pre-trained model (skips training).
        * Evaluate the loaded model on the test set and print the accuracy.
        * *(To force reprocessing or retraining, delete `data/data.npz` or `data/model.h5` respectively).*

3.  **Predicting a Single Image (Optional):**
    * Uncomment the last three lines in the script:
        ```python
        # img = read_image('data\\images\\test_images\\img.png', (128, 128))
        # img = tf.reshape(img, (1 , 128, 128, 1))
        # print('cat' if tf.argmax(nn.predict(img)[0]) == 0 else 'dog')
        ```
    * Make sure you have an image at the specified path (e.g., `data/images/test_images/img.png`).
    * Run the script again. After loading/training and evaluating, it will predict whether the single image is a cat or a dog. *(Note: The script assumes 'cat' corresponds to class index 0 and 'dog' to class index 1 based on the folder reading order).*
    * Because CNNs catch pattens you can also use images of human faces to see if they look more like a dog or a cat.    

## 🧠 How It Works

1.  **Data Loading & Preprocessing (`read_image`, `read_data`, `create_data_npz`):**
    * Images are opened using Pillow.
    * Converted to grayscale (`'L'`).
    * Resized to `(128, 128)` pixels.
    * Converted to NumPy arrays.
    * *Note:* `read_image` flattens the array, but it's immediately reshaped back to `(128, 128, 1)` before training to fit the CNN's input layer.
    * Labels (`y`) are assigned based on the folder index (0, 1).
    * Data is shuffled and saved to `data.npz` for quick loading later.

2.  **Data Splitting:**
    * The loaded data (`X`, `y`) is split into training and testing sets (80/20 split).
    * Labels (`y`) are converted to one-hot encoding using `to_categorical`.

3.  **Model Building (`train_model`):**
    * A `Sequential` Keras model is defined.
    * It consists of three blocks of `Conv2D` (with ReLU activation) and `MaxPooling2D` layers to extract features at different scales.
    * A `Flatten` layer converts the 2D feature maps into a 1D vector.
    * Several `Dense` (fully connected) layers with `Dropout` (for regularization) process the features.
    * The final `Dense` layer has 2 units (one for each class) and a `sigmoid` activation. *(Note: For multi-class classification with `categorical_crossentropy` loss, `softmax` activation is more conventional than `sigmoid` here. For binary classification, either a single output neuron with `sigmoid` and `binary_crossentropy` loss, or two output neurons with `softmax` and `categorical_crossentropy` loss is standard practice).*
    * The model is compiled with the Adam optimizer and `categorical_crossentropy` loss.

4.  **Training & Evaluation (`train_model`, `test_model`):**
    * The model is trained using `neural_net.fit()` on the training data for 25 epochs.
    * The trained model is saved.
    * The `test_model` function predicts classes for the test set, compares them to the true labels, and calculates the accuracy.

---

You can get the data on Kaggle

---

Happy Classifying! 🎉
