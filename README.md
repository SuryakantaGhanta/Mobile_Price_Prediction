# Mobile_Price_Prediction
### Introduction

In the ever-evolving market of mobile phones, predicting the price range of a new device can be a complex yet crucial task. With the diverse range of features and specifications that each phone offers, it's important to understand how these attributes impact the overall pricing. This project aims to build a predictive system that can classify the price range of mobile phones into categories such as low, medium, high, and very high using a dataset of available phones in the market.

By exploring the data, we will gain insights into the significant features that influence mobile pricing. We will leverage various data preprocessing techniques, visualization tools, and machine learning models to develop a robust predictive system. The ultimate goal is to provide a tool that can assist manufacturers, retailers, and consumers in understanding and anticipating the price trends of mobile phones based on their features.

In this project, we will follow a systematic approach:
1. **Data Loading**: Import and load the dataset containing mobile phone features and their corresponding price ranges.
2. **Data Inspection**: Explore and inspect the data to understand its structure, identify missing values, and gain preliminary insights.
3. **Data Preprocessing**: Perform necessary preprocessing steps such as handling missing values, encoding categorical variables, and scaling features.
4. **Feature Engineering and Selection**: Identify and select the most relevant features that contribute to the pricing of mobile phones.
5. **Model Development**: Implement various machine learning models to predict the price range and evaluate their performance.
6. **Evaluation and Analysis**: Assess the performance of the models using appropriate evaluation metrics and analyze the results to draw meaningful conclusions.

By the end of this project, we aim to develop an accurate and efficient system for mobile price prediction, helping stakeholders make informed decisions in the competitive mobile phone market.

### Project Development

#### 1. Required Libraries

In this project, several libraries are essential for data manipulation, preprocessing, model building, and evaluation. Let's go through each of them and understand their significance:

```python
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
```

1. **Pandas (`pd`)**:
   - **Purpose**: Pandas is a powerful library for data manipulation and analysis. It provides data structures like DataFrames, which allow for efficient data handling and operations.
   - **Usage**: We use Pandas to load the dataset, inspect it, and perform various data preprocessing tasks.

2. **Scikit-Learn (`sk`)**:
   - **Purpose**: Scikit-Learn is a comprehensive machine learning library that provides tools for data preprocessing, model building, and evaluation.
   - **Usage**: We leverage Scikit-Learn for splitting the data into training and testing sets, scaling features, implementing machine learning models, and evaluating their performance.

3. **NumPy (`np`)**:
   - **Purpose**: NumPy is a fundamental library for numerical computing in Python. It provides support for arrays, matrices, and a wide range of mathematical functions.
   - **Usage**: NumPy is used for handling numerical data and performing mathematical operations that are essential for data preprocessing and model development.

4. **MinMaxScaler**:
   - **Purpose**: This scaler is part of the Scikit-Learn library and is used to scale features to a given range (default is 0 to 1).
   - **Usage**: We use MinMaxScaler to normalize the features, ensuring that they are on a similar scale, which can improve the performance of machine learning models.

5. **train_test_split**:
   - **Purpose**: This function, provided by Scikit-Learn, is used to split the dataset into training and testing sets.
   - **Usage**: Splitting the data allows us to train the model on one subset of the data and test its performance on another subset, ensuring that the model generalizes well to unseen data.

6. **RandomizedSearchCV**:
   - **Purpose**: This is a hyperparameter tuning technique provided by Scikit-Learn. It performs random searches over specified parameter values.
   - **Usage**: We use RandomizedSearchCV to find the best hyperparameters for our model, improving its performance and accuracy.

7. **RandomForestClassifier**:
   - **Purpose**: This is an ensemble learning method provided by Scikit-Learn, which combines multiple decision trees to improve classification performance.
   - **Usage**: We use RandomForestClassifier to build a robust and accurate model for predicting the price range of mobile phones.

8. **classification_report & confusion_matrix**:
   - **Purpose**: These metrics, provided by Scikit-Learn, are used to evaluate the performance of classification models.
   - **Usage**: The classification report provides precision, recall, and F1-score, while the confusion matrix gives a detailed breakdown of correct and incorrect predictions, helping us assess the model's performance.

These libraries and functions form the backbone of our project, enabling us to efficiently handle data, build predictive models, and evaluate their performance accurately.

#### 2. Data Loading

Loading the dataset is the first crucial step in any data science project. Here's how we load the dataset and take a quick look at the initial rows:

```python
m_data = pd.read_csv(r'C:\Users\USER\Documents\mobile_phone_pricing\Mobile Phone Pricing\dataset.csv')
m_data.head()
```

**Explanation:**
- **Purpose**: This code snippet loads the dataset containing mobile phone features and their corresponding price ranges into a Pandas DataFrame named `m_data`.
  - `pd.read_csv(r'C:\Users\USER\Documents\mobile_phone_pricing\Mobile Phone Pricing\dataset.csv')` reads the CSV file from the specified path and loads it into a DataFrame.
  - `m_data.head()` displays the first few rows of the dataset to give an initial overview of the data.
- **Reason**: Loading the data into a DataFrame allows for easy data manipulation and analysis. Viewing the first few rows helps in understanding the structure and content of the dataset, which is essential for subsequent data preprocessing and modeling steps.

#### 3. Data Inspection and Wrangling

Once the data is loaded, inspecting it is the next step. This helps us understand the data better and identify any issues such as missing values or outliers. Here are the commands for inspecting the dataset:

```python
m_data.describe()
m_data.isnull().sum()
m_data.nunique()
```

**Explanation:**
- **Purpose**: These commands are used to inspect the dataset and gather essential information:
  - `m_data.describe()`: Generates descriptive statistics of numerical columns, providing insights into the data distribution, central tendency, and spread.
  - `m_data.isnull().sum()`: Checks for missing values in the dataset by counting the number of null entries in each column.
  - `m_data.nunique()`: Counts the number of unique values in each column, which helps in understanding the diversity of data in each feature.
- **Reason**: Inspecting the data helps us identify any potential issues that need to be addressed during the data preprocessing stage. For instance, detecting missing values and understanding the distribution of data can guide us in choosing appropriate preprocessing techniques.

Identifying columns with binary data is crucial for effective feature engineering and modeling. Here's a function to identify binary columns in the dataset:

```python
m_data.columns
def identify_binary_columns(df):
    binary_columns = []
    for column in df.columns:
        n_unique = df[column].nunique()
        if n_unique == 2:
            binary_columns.append(column)
            print(f"Binary column: {column}")
            print(f"Unique values: {df[column].unique()}")
            print("-" * 40)
    return binary_columns

# Get binary columns
binary_cols = identify_binary_columns(m_data)

# Print summary
print(f"\nTotal binary columns found: {len(binary_cols)}")
print("\nBinary columns list:")
print(binary_cols)

nb_cols = [col for col in m_data.columns if col not in binary_cols]

print("Non-binary columns:", nb_cols)
```

**Explanation:**
- **Purpose**: This code snippet identifies and lists binary columns in the dataset.
  - `identify_binary_columns(df)`: A function that iterates over each column in the DataFrame and checks the number of unique values. If a column has exactly two unique values, it is considered binary.
  - `binary_columns.append(column)`: Adds the binary column to the list `binary_columns`.
  - The function prints the name and unique values of each binary column for inspection.
  - The function returns the list of binary columns.
- **Reason**: Identifying binary columns helps in understanding the categorical nature of the data and allows for specific preprocessing steps tailored to binary features. This is important for accurate modeling and interpretation.

Scaling and transforming the non-binary columns is the next step:

```python
cols_to_scale = list(set(nb_cols) - set(['price_range']))
scaler = MinMaxScaler()
m_data[cols_to_scale] = scaler.fit_transform(m_data[cols_to_scale])
m_data.head()
```

**Explanation:**
- **Purpose**: This code snippet scales the non-binary columns to a range of 0 to 1 using MinMaxScaler.
  - `MinMaxScaler` is used to normalize the features, ensuring that they are on a similar scale.
  - `m_data[cols_to_scale] = scaler.fit_transform(m_data[cols_to_scale])`: Applies the scaler to the specified columns and updates the DataFrame.
- **Reason**: Scaling features can improve the performance of machine learning models by ensuring that all features contribute equally to the model training.

Finally, we prepare the data for model training by defining the features (`X`) and target variable (`y`) and splitting the data into training and testing sets:

```python
X = m_data.drop('price_range', axis=1)
y = m_data['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Explanation:**
- **Purpose**: This code snippet prepares the data for model training.
  - `X` represents the features, while `y` represents the target variable (`price_range`).
  - `train_test_split` is used to split the data into training and testing sets, with 20% allocated for testing.
- **Reason**: Splitting the data into training and testing sets allows for evaluating the performance of the model on unseen data, ensuring that it generalizes well.

#### 4. Model Development and Feature Importance


```python
# Create base model
rf = RandomForestClassifier(random_state=42)
```
- **Purpose**: This line initializes a RandomForestClassifier model with a fixed random state to ensure reproducibility.

```python
# Perform RandomizedSearchCV
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
```
- **Purpose**: This block sets up the hyperparameter tuning process using `RandomizedSearchCV`:
  - **`estimator=rf`**: Specifies the base model to be tuned.
  - **`param_distributions=param_grid`**: Provides the grid of hyperparameters to search over.
  - **`n_iter=100`**: Indicates that 100 different combinations of hyperparameters will be tried.
  - **`cv=5`**: Uses 5-fold cross-validation to evaluate each combination.
  - **`verbose=2`**: Controls the verbosity level of output during the search.
  - **`random_state=42`**: Ensures reproducibility of the search results.
  - **`n_jobs=-1`**: Utilizes all available CPU cores for parallel processing.

```python
# Fit the random search model
rf_random.fit(X_train, y_train)
```
- **Purpose**: Fits the `RandomizedSearchCV` model on the training data (`X_train` and `y_train`). This process tunes the hyperparameters and identifies the best combination based on the cross-validation performance.

```python
# Print best parameters
print("Best Parameters:", rf_random.best_params_)
```
- **Purpose**: Prints the best hyperparameters found during the random search.

```python
# Get best model
best_model = rf_random.best_estimator_
```
- **Purpose**: Extracts the best model (i.e., RandomForestClassifier with the best hyperparameters) from the `RandomizedSearchCV` results.

```python
# Make predictions
y_pred = best_model.predict(X_test)
```
- **Purpose**: Uses the best model to make predictions on the test data (`X_test`).

```python
# Print model evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```
- **Purpose**: Prints the classification report, which includes precision, recall, F1-score, and support for each class. This provides a detailed evaluation of the model's performance.

```python
# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
```
- **Purpose**: Creates a data frame to store the feature importance values. It sorts the features in descending order of importance.

```python
# Print top 10 important features
print("\nTop 10 Important Features:")
print(feature_importance.head(10))
```
- **Purpose**: Prints the top 10 most important features, highlighting the features that contribute the most to the model's predictions.

This code snippet performs hyperparameter tuning, evaluates the model, and identifies the most important features for predicting mobile phone price ranges. 
### Evaluation and Conclusion

**Classification Report:**
- The classification report provides a detailed evaluation of the model's performance. The key metrics include precision, recall, and F1-score for each class (0, 1, 2, and 3, representing low, medium, high, and very high price ranges, respectively).
  - **Precision**: Measures the accuracy of the positive predictions.
  - **Recall**: Measures the ability to capture all positive instances.
  - **F1-score**: Harmonic mean of precision and recall, providing a balance between the two.
- The support column indicates the number of actual occurrences for each class in the test set.
- The overall accuracy of the model is 89%, indicating that the model correctly predicts the price range for 89% of the test samples.
- The macro average and weighted average provide an overall summary of the model's performance across all classes, with the macro average treating all classes equally, and the weighted average taking class imbalance into account.

**Top 10 Important Features:**
- The feature importance scores indicate the contribution of each feature to the model's predictions.
  - **RAM (0.552759)**: The most important feature, significantly impacting the price range prediction.
  - **Battery Power (0.073262)**: Second most important feature, highlighting the importance of battery capacity.
  - **Pixel Height (0.051545) and Pixel Width (0.051034)**: Resolution features that contribute to the perceived value of the mobile phone.
  - **Mobile Weight (0.033175)**: The weight of the mobile phone affects its overall build quality and pricing.
  - **Internal Memory (0.031497)**: Storage capacity, which is a key factor in determining the price.
  - **Talk Time (0.024072)**: Battery performance measure that influences pricing.
  - **Primary Camera (PC) (0.023368)**: Camera quality, which is a significant feature for users.
  - **Screen Height (SC_H) (0.023007) and Screen Width (SC_W) (0.022572)**: Screen size dimensions that affect the user experience and pricing.

**Conclusion:**
- The RandomForestClassifier model achieved an overall accuracy of 89%, demonstrating its effectiveness in predicting the price range of mobile phones.
- The classification report shows that the model performs well across all classes, with high precision, recall, and F1-scores.
- Feature importance analysis reveals that RAM, battery power, pixel dimensions, and internal memory are the most significant features influencing the price range prediction.
- These insights can guide manufacturers and retailers in understanding the key factors that drive mobile phone pricing and help in making informed decisions. The model provides a reliable tool for predicting the price range of new mobile phones based on their features.
