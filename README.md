# AI vs Human Text Classifier

This project builds a deep learning text classifier using TensorFlow/Keras to distinguish between AI-generated and human-generated essays. The dataset (available on [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)) consists of a CSV file with two columns:

- **text**: The essay.
- **generated**: A numeric label where `0` indicates human-generated text and `1` indicates AI-generated text.

**Note:** This project uses a simple CNN-based model for demonstration. For improved performance, you may experiment with more advanced architectures (e.g., LSTM/GRU or Transformer-based models).

## Table of Contents

- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)

## Installation

This notebook is designed to run in Google Colab. The required packages are installed automatically in the notebook. The following packages are used:
- TensorFlow
- Pandas
- Scikit-learn
- Kaggle (for dataset download)

## Dataset Setup

Instead of uploading a large CSV (1.1GB) manually, the notebook uses the Kaggle CLI to download the dataset directly. Make sure you have your Kaggle API credentials set up:
1. Go to your Kaggle account and download your `kaggle.json`.
2. In Colab, upload the `kaggle.json` file and copy it into the `~/.kaggle/` folder (the notebook includes these commands).

The dataset slug and CSV file name are specified in the code. In this example, we use the dataset slug `"shanegerami/ai-vs-human-text"` and limit the data to the first 50,000 rows for faster training and lower memory usage.

## Usage

### Training the Model

1. **Download the Dataset:**  
   The notebook uses Kaggle CLI to download and unzip the dataset automatically.

2. **Load and Preprocess Data:**  
   The CSV is read into a Pandas DataFrame and the data is limited (e.g., to 50K rows). The `generated` column is converted to an integer type. A train/validation split is performed.

3. **Text Vectorization:**  
   A Keras `TextVectorization` layer converts raw text into sequences of integers, with a maximum vocabulary size and fixed sequence length.

4. **Build tf.data Pipelines:**  
   The data is wrapped into TensorFlow datasets with batching, shuffling, and prefetching.

5. **Define the Model:**  
   A simple CNN model (with Embedding, Conv1D, GlobalMaxPooling1D, Dense layers, and Dropout) is constructed to classify the texts.

6. **Model Training:**  
   The model is compiled and trained using the training dataset, with validation on a hold-out set.

### Evaluation

After training, the model is evaluated on the validation set to obtain loss and accuracy metrics.

### Prediction

The notebook includes a helper function to predict whether new text is AI-generated or human-generated. For each new text, the function:
- Vectorizes the text.
- Uses the trained model to predict a probability.
- Returns a label (1 for AI, 0 for Human) based on a threshold (0.5).
