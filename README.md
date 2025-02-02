# Number-prediction
prediction model using pytorch 
## Overview
This project focuses on predicting handwritten digits using machine learning. The dataset used for training and testing is the MNIST dataset, which contains labeled images of digits (0-9). The goal is to build a model that accurately classifies each digit.

## Dataset
The MNIST dataset consists of:
- **60,000 training images**
- **10,000 testing images**
- Each image is a **28x28 grayscale pixel array**

## Project Goals
1. **Data Preprocessing**:
   - Normalization and reshaping of images
   - Splitting data into training and testing sets

2. **Exploratory Data Analysis (EDA)**:
   - Visualizing sample images
   - Understanding class distribution

3. **Model Training**:
   - Building a neural network using PyTorch
   - Experimenting with different architectures (CNN, MLP)

4. **Model Evaluation**:
   - Accuracy and loss analysis
   - Confusion matrix for error analysis

## Tools & Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **PyTorch** (for deep learning models)
- **Jupyter Notebook**
- **Scikit-Learn** (for preprocessing and evaluation)

## Installation
To run this project, clone the repository and install the dependencies:

```bash
git clone https://github.com/Apurv-panbude24/Number-prediction.git
cd Number-prediction
pip install -r requirements.txt
```

## Usage
1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Run the notebooks step by step to train and evaluate the model.

## Results & Insights
- **Achieved over 98% accuracy on test data**
- **CNN outperforms traditional MLP models**
- **Misclassified images mainly involve digits with similar shapes**

## Future Enhancements
- Implement data augmentation for better generalization
- Experiment with more advanced architectures (e.g., ResNet)
- Deploy the model as a web application

## Contributors
- **Apurv Panbude** ([@Apurv-panbude24](https://github.com/Apurv-panbude24))

## License
This project is licensed under the MIT License.

