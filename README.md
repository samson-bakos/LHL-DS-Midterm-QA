# Data Science Midterm Demo

Demo version of the Midterm Assignment for LHL DS 5.0 Curriculum

Demo by: Samson Bakos

# Example Read Me 

## Data Science Midterm Project - Housing Price Prediction

Welcome to the Data Science Midterm Project on Housing Price Prediction. This project involves a comprehensive approach to predicting housing prices in the US using various data science techniques. The project is structured into three main parts: Exploratory Data Analysis (EDA), Model Selection, and Tuning & Pipelining. Below is a detailed breakdown of the project's structure, tasks, and objectives.

## Project Goals
The goal of this project is to develop a robust model to predict housing prices based on various features extracted from housing data. The project aims to leverage different data science methodologies including data preprocessing, model selection, and model tuning to achieve the best possible predictions.

## Process

### Part 1: Exploratory Data Analysis (EDA)
- **Data Loading**: Load the housing data from multiple JSON files located in the `data/` directory into Pandas dataframes.
- **Data Cleaning and Preprocessing**: Explore, clean, and preprocess the data to make it suitable for ML modeling. 
- **External Data Sources (Stretch Goal)**: Investigate potential external data sources that could enhance the model's predictive power. 
- **Data Exploration**: Conduct thorough exploratory data analysis to understand the distributions and relationships among the various features.
- **Data Saving**: Save the processed datasets into the `data/processed/` subfolder as CSV files.

### Part 2: Model Selection
- **Model Experimentation**: Apply various supervised learning models on the preprocessed data.
- **Model Selection Criteria**: Define and justify the criteria for selecting the best model based on performance metrics relevant to housing price predictions.
- **Feature Selection (Stretch Goal)**: Explore the necessity of each feature in the dataset and its impact on model performance. 

### Part 3: Tuning and Pipelining
- **Hyperparameter Tuning**: Tune the hyperparameters of the best performing models to enhance their predictions. Address data leakage issues during preprocessing by creating custom optimization functions (Stretch Goal). 
- **Model Saving**: Save the best-tuned model in a newly created `models/` directory.
- **Pipeline Construction** (Stretch Goal): Build a pipeline that incorporates preprocessing steps and utilizes the tuned model to predict new data. Save this pipeline. 

## Results
An XGBoost model was selected. Before tuning, this model yielded an $R^2$ score of 0.998, while the final tuned model had an $R^2$ of 0.994, indicating an extremely high level of predictive power. 

Despite the marginally lower $R^2$ score in the tuned model, this model was selected due to a significantly lower MAE (mean error in actual dollars), achieving an MAE of $\pm 3756$, decreased from $\pm 8710$ in the untuned model. 

## Challenges

### Preventing Data Leakage during Hyperparameter Tuning

One primary challenge in this project arises from the method used to encode categorical data. Specifically, the encoding of the 'city' feature was done by calculating the mean of the sold prices for each city based on the entire training dataset. This presents a significant issue for model validation and hyperparameter tuning. 

Traditionally, tools like `GridSearchCV` from Scikit-Learn are employed to optimize hyperparameters while performing cross-validation to prevent overfitting. However, using this approach without modification in our case leads to data leakage. This is because the mean sold prices (used to encode the cities) include data from all training observations, some of which would end up in the validation fold of each cross-validation split. Thus, information from the validation data inadvertently influences the training process, which can falsely inflate the model's performance metrics and lead to a model that may not perform well on genuinely unseen data.

To address this challenge, custom functions for cross-validation and hyperparameter tuning were necessary. These functions ensure that the city means are recalculated for each training fold separately, thereby preserving the integrity of the validation process and ensuring that our performance metrics are a true reflection of the model's ability to generalize to new data.

## Future Goals
- **Further Data Integration**: Explore additional datasets and features that could improve model accuracy.
- **Feature Selection**: Explore algorithmic feature selection to create a lower dimensionality, more performative model.
- **Advanced Models**: Experiment with more advanced machine learning and deep learning models.
- **Deployment**: Set up an API for real-time housing price predictions based on the developed models.


