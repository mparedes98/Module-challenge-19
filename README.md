# Employee Attrition Prediction with Branched Neural Network

This project implements a branched neural network to predict both employee attrition (whether employees will leave the company) and department assignment using TensorFlow/Keras. The model uses a shared feature extraction architecture with two specialized output branches.

## Overview

The branched neural network architecture allows simultaneous prediction of two related tasks:
- **Attrition Prediction**: Binary classification (Will employee leave: Yes/No)
- **Department Prediction**: Multi-class classification (Which department: Sales, R&D, HR)

This multi-task learning approach can improve model performance by sharing learned representations between related prediction tasks.

## Dataset

The employee attrition dataset contains **1,470 employee records** with 27 features including:

### Selected Features (10 used in model):
- **Age**: Employee age
- **Education**: Education level (1-5 scale)
- **DistanceFromHome**: Distance from home to workplace
- **JobSatisfaction**: Job satisfaction rating (1-4 scale)
- **OverTime**: Whether employee works overtime (Yes/No)
- **StockOptionLevel**: Stock option level (0-3)
- **WorkLifeBalance**: Work-life balance rating (1-4 scale)
- **YearsAtCompany**: Years employed at company
- **YearsSinceLastPromotion**: Years since last promotion
- **NumCompaniesWorked**: Number of previous companies

### Target Variables:
- **Attrition**: Employee turnover (Yes/No)
- **Department**: Employee department (Sales, Research & Development, Human Resources)

## Model Architecture

### Network Structure
```
Input Layer (10 features)
    ↓
Shared Layer 1 (64 neurons, ReLU)
    ↓
Shared Layer 2 (128 neurons, ReLU)
    ↓
    ├─── Department Branch                 ├─── Attrition Branch
    │    ├─ Hidden (32 neurons, ReLU)      │    ├─ Hidden (32 neurons, ReLU)
    │    └─ Output (3 neurons, Softmax)    │    └─ Output (1 neuron, Sigmoid)
    │                                      │
    └─── Department Prediction             └─── Attrition Prediction
```

### Key Components
- **Shared Layers**: Extract common features useful for both predictions
- **Department Branch**: Multi-class classification with softmax activation
- **Attrition Branch**: Binary classification with sigmoid activation
- **Total Parameters**: ~17,445 trainable parameters

## Data Preprocessing

### Feature Engineering
1. **Categorical Encoding**: Convert OverTime (Yes/No → 1/0)
2. **Feature Scaling**: StandardScaler normalization (mean=0, std=1)
3. **Target Encoding**: One-hot encoding for both output variables
4. **Train/Test Split**: 80% training, 20% testing

### Preprocessing Pipeline
```python
# Data splitting
train_test_split(test_size=0.2, random_state=42)

# Feature scaling
StandardScaler() fitted on training data only

# Target encoding
OneHotEncoder() for Department (3 categories)
OneHotEncoder() for Attrition (2 categories)
```

## Model Performance

### Training Configuration
- **Optimizer**: Adam
- **Loss Functions**: 
  - Department: Categorical crossentropy
  - Attrition: Binary crossentropy
- **Metrics**: Accuracy for both outputs
- **Epochs**: 100
- **Batch Size**: 32

### Expected Performance
- **Department Accuracy**: ~50-70%
- **Attrition Accuracy**: ~78-85%
- **Training Time**: ~30 seconds (100 epochs)

## File Structure

```
neural-network-challenge-2/
├── attrition.ipynb              # Main implementation notebook
├── attrition.csv               # Employee dataset
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

## Usage

### Requirements
```bash
pip install tensorflow pandas scikit-learn numpy
```

### Running the Model

1. **Load and Explore Data**:
   ```python
   attrition_df = pd.read_csv('attrition.csv')
   attrition_df.head()
   ```

2. **Preprocess Data**:
   ```python
   # Create features and targets
   X_df = attrition_df[selected_features]
   y_df = attrition_df[['Attrition', 'Department']]
   
   # Split and scale data
   X_train, X_test, y_train, y_test = train_test_split(...)
   X_train_scaled = scaler.fit_transform(X_train)
   ```

3. **Build Model**:
   ```python
   # Create branched architecture
   model = Model(inputs=input_layer, 
                outputs=[department_output, attrition_output])
   ```

4. **Train and Evaluate**:
   ```python
   # Train model
   model.fit(X_train_scaled, {'department_output': y_train_dept, 
                             'attrition_output': y_train_attr}, 
            epochs=100)
   
   # Evaluate performance
   results = model.evaluate(X_test_scaled, {...})
   ```

## Technologies Used

- **Python 3.x**
- **TensorFlow/Keras**: Neural network framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Preprocessing and model evaluation
- **NumPy**: Numerical computations

## Model Insights

### Activation Function Choices
- **Softmax (Department)**: Ensures probabilities sum to 1 for multi-class prediction
- **Sigmoid (Attrition)**: Outputs probability between 0-1 for binary classification

### Loss Function Rationale
- **Categorical Crossentropy**: Optimal for multi-class problems with one-hot encoded targets
- **Binary Crossentropy**: Standard choice for binary classification tasks

## Limitations and Improvements

### Current Limitations
- **Class Imbalance**: Attrition data likely skewed toward "No" (employees staying)
- **Limited Features**: Only 10 of 27 available features used
- **Accuracy Metric**: May not be ideal for imbalanced attrition prediction

### Potential Improvements
1. **Advanced Metrics**: Use precision, recall, F1-score, AUC-ROC for attrition
2. **Class Balancing**: Implement SMOTE or class weights for imbalanced data
3. **Feature Engineering**: Create interaction terms and derived features
4. **Regularization**: Add dropout layers to prevent overfitting
5. **Hyperparameter Tuning**: Optimize architecture and training parameters
6. **Ensemble Methods**: Combine multiple models for improved performance

## Future Enhancements

- **Feature Importance Analysis**: Identify key predictors using SHAP or permutation importance
- **Cross-Validation**: Implement k-fold CV for robust performance estimation
- **Model Interpretability**: Add visualization for model decisions and feature impacts
- **Real-time Prediction**: Deploy model as REST API for production use
- **A/B Testing Framework**: Compare against baseline models and business rules

## Code Source

This project was developed as part of a neural network bootcamp assignment, implementing best practices for:
- Multi-task learning with branched architectures
- Proper data preprocessing and feature engineering
- TensorFlow/Keras functional API usage
- Model evaluation and performance analysis

## License

This project is for educational purposes as part of a machine learning bootcamp assignment.

## Contact

For questions about implementation details or model improvements, please refer to the course materials or contact the development team.

## Code Source

All code was developed for this educational project following standard machine learning practices for binary classification with neural networks.
