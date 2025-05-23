# Student Loan Risk Prediction with Deep Learning

This project uses a deep neural network to predict student loan credit risk based on various student and loan characteristics.

## Overview

The model predicts whether a student will have a good credit ranking (0 or 1) for loan repayment based on features such as payment history, GPA, academic performance, and financial factors.

## Dataset

The dataset contains 1,599 student loan records with the following features:
- `payment_history`: Student's payment track record
- `location_parameter`: Geographic factor
- `stem_degree_score`: STEM degree indicator
- `gpa_ranking`: Academic performance ranking
- `alumni_success`: Alumni network success metric
- `study_major_code`: Academic major code
- `time_to_completion`: Time to degree completion
- `finance_workshop_score`: Financial literacy score
- `cohort_ranking`: Peer group ranking
- `total_loan_score`: Overall loan assessment
- `financial_aid_score`: Financial aid evaluation
- `credit_ranking`: Target variable (0 = poor, 1 = good)

## Model Architecture

- **Input Layer**: 11 features
- **Hidden Layer 1**: 6 neurons with ReLU activation
- **Hidden Layer 2**: 3 neurons with ReLU activation  
- **Output Layer**: 1 neuron with sigmoid activation (binary classification)
- **Total Parameters**: 97

## Model Performance

- **Test Accuracy**: 73.5%
- **Test Loss**: 0.505
- **Training**: 50 epochs with binary crossentropy loss and Adam optimizer

## Classification Results

```
              precision    recall  f1-score   support
           0       0.70      0.76      0.73       188
           1       0.77      0.72      0.74       212
    accuracy                           0.73       400
   macro avg       0.74      0.74      0.73       400
weighted avg       0.74      0.73      0.74       400
```

## Files

- `student_loans_with_deep_learning.ipynb`: Main Jupyter notebook with analysis
- `student_loans.keras`: Saved trained model
- `README.md`: This file

## Technologies Used

- Python 3.x
- TensorFlow/Keras
- Pandas
- Scikit-learn
- NumPy

## Usage

1. Load the dataset from the provided CSV file
2. Preprocess the data (scaling, train/test split)
3. Train the neural network model
4. Evaluate performance on test data
5. Make predictions on new student loan applications

## Recommendation System Discussion

The project also explores building a recommendation system for student loans using content-based filtering, considering challenges like regulatory compliance and long-term financial impact.

## Code Source

All code was developed for this educational project following standard machine learning practices for binary classification with neural networks.
