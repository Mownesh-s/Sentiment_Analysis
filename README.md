# Sentiment Analysis with Deep Learning

## Project Overview
This project implements a sentiment analysis model using deep learning. It classifies text data as positive or negative using an LSTM-based neural network trained on a dataset from the `datasets` library.

## Features
- Uses TensorFlow and Keras for deep learning model implementation.
- Processes dataset from the `datasets` library.
- Includes data preprocessing steps such as tokenization and normalization.
- Uses an LSTM-based model for sentiment classification.
- Evaluates model performance using accuracy and loss metrics.
- Visualizes training progress using accuracy and loss curves.
- Generates predictions on new text inputs.

## Dataset
The dataset is sourced from the `datasets` library and contains labeled text data for sentiment classification. It consists of:
- Positive and negative sentiment labels.
- A large collection of text samples for training and evaluation.

## Installation
To run this project, install the required dependencies:
```bash
pip install tensorflow datasets numpy pandas matplotlib
```

## Usage
Run the Jupyter Notebook to execute the sentiment analysis model step by step.
```bash
jupyter notebook Sentiment_Analysis.ipynb
```

## Data Preprocessing
- The dataset is loaded and split into training and testing sets.
- Text preprocessing includes:
  - Tokenization
  - Stopword removal
  - Normalization
  - Padding sequences

## Model Architecture
The deep learning model consists of:
- An **embedding layer** for text representation.
- A **bidirectional LSTM** layer for capturing context.
- Fully connected layers for classification.

## Model Training
- The dataset is preprocessed and transformed into a suitable format.
- The model is compiled with an optimizer and loss function.
- The training process is visualized using real-time accuracy and loss plots.

### Training Visualization
The following plots help understand model performance:
- **Accuracy Curve:** ![Accuracy Curve](images/accuracy_curve.png)
- **Loss Curve:** ![Loss Curve](images/loss_curve.png)

## Evaluation Metrics
The model's performance is assessed using:
- Accuracy
- Loss

## Results
After training, the model achieves competitive accuracy in classifying sentiments. The loss curve helps in understanding model optimization.

## Making Predictions
Use the trained model to predict sentiment for new text inputs:
```python
text = "I love this product! It's amazing."
prediction = model.predict([text])
print("Predicted Sentiment:", "Positive" if prediction > 0.5 else "Negative")
```

## Future Improvements
- Implement hyperparameter tuning.
- Experiment with more complex architectures like Transformers.
- Deploy the model as a web application.
- Fine-tune the model with additional datasets.

