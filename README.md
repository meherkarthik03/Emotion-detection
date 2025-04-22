# AI-Based Emotion Detection from Physiological Signals

This project focuses on emotion classification using physiological signals with multiple machine learning and deep learning models. 
It implements and compares models such as Convolutional Neural Networks (CNN), Random Forest Classifier, and Naive Bayes to identify emotional states from processed biosignal data.


## Project Objectives

- To build and compare multiple models (CNN, Random Forest, Naive Bayes) for emotion classification.
- To preprocess physiological data for model compatibility.
- To visualize and interpret the performance using evaluation metrics.
- To provide a user-friendly web interface using Streamlit for real-time emotion prediction.



##  Models Implemented

| Model              | Description                                     |
|-------------------|-------------------------------------------------|
| CNN (Keras)        | Deep learning model for complex feature learning |
| Random Forest      | Tree-based ensemble learning method             |
| Naive Bayes        | Probabilistic classifier based on Bayes' theorem|



## Technologies Used

- **Languages**: Python 3.10+
- **Frameworks**: TensorFlow, Keras, scikit-learn
- **Interface**: Streamlit
- **Visualization**: Seaborn, Matplotlib
- **Model Evaluation**: Confusion Matrix, Accuracy, Classification Report



## Dataset

The dataset used includes physiological features (e.g., ECG, EDA, EMG) and their corresponding emotional labels. It is preprocessed and scaled using `StandardScaler`, then split for training and testing.


## Performance Evaluation

- Accuracy, precision, recall, and F1-score are used to compare model effectiveness.
- CNN shows improved performance over classical ML models like Random Forest and Naive Bayes.
- Visualizations like confusion matrices provide insight into model behavior.





### 1. Clone the Repository

```bash
git clone https://github.com/meherkarthik03/emotion-classification-ai.git
cd emotion-classification-ai
