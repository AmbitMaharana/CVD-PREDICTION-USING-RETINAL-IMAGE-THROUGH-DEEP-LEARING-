# CVD Prediction Using Retinal Images Through Deep Learning

## Overview

This project aims to predict cardiovascular disease (CVD) risk using retinal images through a deep learning model. The model is built using a Convolutional Neural Network (CNN) trained on retinal images labeled with disease risk.

## Features

- Uses retinal fundus images for CVD risk prediction.
- Implements a deep learning model using TensorFlow and Keras.
- Includes model training, evaluation, and real-time prediction capabilities.
- Provides data visualization with accuracy and loss plots.

## Dataset

The dataset consists of retinal fundus images and a corresponding CSV file containing image IDs and risk labels. The dataset is sourced from Kaggle.

## Usage

### Running the Model in Google Colab

Since the project is designed to run on Google Colab with a Kaggle dataset, follow these steps:

1. **Open Google Colab**
2. **Upload `main.py`** or clone the repository into Colab:
   ```sh
   !git clone https://github.com/yourusername/CVD-PREDICTION-USING-RETINAL-IMAGE.git
   cd CVD-PREDICTION-USING-RETINAL-IMAGE
   ```
3. **Set up Kaggle API key** (upload your `kaggle.json` file to Colab and use it to download the dataset).
4. **Run the script**:
   ```sh
   !python main.py
   ```

## Model Architecture

The model consists of:

- Convolutional layers (Conv2D) for feature extraction.
- MaxPooling layers for downsampling.
- Fully connected Dense layers for classification.
- Sigmoid activation for binary classification.

## Results

The model achieves an accuracy of approximately **88%** on validation data. Below are the accuracy and loss plots during training:

![Training Accuracy and Loss](images/plot.png)


## Folder Structure

```
CVD-PREDICTION-USING-RETINAL-IMAGE/
│── main.py                # Main script for training and prediction
│── data/                  # Dataset folder (Kaggle dataset)
│── models/                # Saved models
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## Dependencies

Install dependencies from `requirements.txt`:

```
tensorflow
pandas
numpy
matplotlib
scikit-learn
google-colab
kaggle
```

## Contributing

Feel free to contribute by submitting issues or pull requests.

## License

This project is licensed under the MIT License.

## Git Code

### Steps to Upload to GitHub

1. **Initialize Git and Add Files**

   ```sh
   git init
   git add .
   git commit -m "Initial commit - CVD Prediction Project"
   ```

2. **Create a New Repository on GitHub**  
   Go to [GitHub](https://github.com/new) and create a repository named `CVD-PREDICTION-USING-RETINAL-IMAGE`.

3. **Connect Local Repo to GitHub**

   ```sh
   git remote add origin https://github.com/yourusername/CVD-PREDICTION-USING-RETINAL-IMAGE.git
   git branch -M main
   git push -u origin main
   ```

4. **Verify the Push** Go to `https://github.com/yourusername/CVD-PREDICTION-USING-RETINAL-IMAGE` to see your files.






