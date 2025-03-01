# Sentiment Analysis with Deep Learning

## Project Overview
This project implements a sentiment analysis model using deep learning. It classifies text data as positive , negative and neutral using an LSTM-based neural network trained on a dataset from the `datasets` library.

## Features
- Uses TensorFlow and Keras for deep learning model implementation.
- Processes dataset from the `datasets` library.
- Includes data preprocessing steps such as tokenization.
- Uses an LSTM-based model for sentiment classification.
- Evaluates model performance using accuracy and loss metrics.
- Visualizes training progress using accuracy and loss curves.
- Generates predictions on new text inputs.

## Dataset
The dataset is "yelp_review_full" which is sourced from the `datasets` library and contains labeled text data for sentiment classification. It consists of:
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
  - Padding sequences

 ## Used GloVe Embedding 
Used pre trained glove embedding to make the model concentrate on classifying the inputs.

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
- **Accuracy & Loss Curve**
  ![image](https://github.com/user-attachments/assets/666d8371-d9db-4d3e-9948-0b2654bb7da9)

## Evaluation Metrics
The model's performance is assessed using:
- Accuracy
- Loss

## Results
After training, the model achieves competitive accuracy in classifying sentiments. The loss curve helps in understanding model optimization.

## Test Accuracy
Model reached a Test Accuracy: 0.8193-(82%)

## Making Predictions
Use the trained model to predict sentiment for new text inputs:
```python
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model.predict(padded)
    labels = ['Negative', 'Neutral', 'Positive']
    return labels[np.argmax(pred)]
```
## Sample Predictions
![image](https://github.com/user-attachments/assets/60a6ab69-d7c7-419c-a0c4-b76073e549b6)

## Future Improvements
- Implement hyperparameter tuning.
- Experiment with more complex architectures like Transformers.
- Deploy the model as a web application.
- Fine-tune the model with additional datasets.

