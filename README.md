# **Building and Deploying an End-to-End Machine Learning Project:   A Predictive Model to Analyse the Academic Performance of Different Ethnic Groups**

*Author : Chandrika Sethumadhavan*


Python version used : 3.12, IDE used: Visual Studio Code


Building a machine learning (ML) project isn't just about developing a model, it's about taking it from inception to deployment in an organized and systematic manner. The process involves several stages, and in this project, I followed the standard ML pipeline to ensure the project's scalability, maintainability, and efficiency. Below is an overview of how I approached the problem step-by-step.

# **1. Problem Understanding and Goal Setting**
Before diving into the development process, it's essential to understand the problem domain. I began by outlining the problem at hand, creating a mindmap of the next steps, and defining the exact goals that I aimed to achieve. This structured approach helped me stay focused on the project's objectives throughout.

# **2. Data Collection and Exploration**
Once the problem and goals were clear, the next step was data collection. In this stage, I worked with the data to ensure it is clean and well-prepared for the next steps.

**2.1 Exploratory Data Analysis (EDA)**:

EDA is crucial to understand the dataset, its distribution, and to identify potential issues such as missing values or outliers. I used various visualization techniques and statistical methods to analyze the dataset. Here are the results of the EDA:

| EDA Task	 | Observations and results   | 
|:---------------|:----------------------------|
| Data health check         | Verified no missing values, inspected datatypes.        | 
| Univariate plots        | Histograms for math_score, reading_score, writing_score revealed near-normal bell curves.       | 
| Feature engineering during EDA         | Created an average column (mean of three scores) and a pass_fail flag (average ≥ 60).         |
| Bivariate analysis |  Boxplots showed standard lunch > free/reduced lunch in every subject. Grouped bar charts confirmed females edge males overall, but males top math. |
| correlation heat-map | Strong positive correlation between all three subject scores, serves helpful later for feature selection.|
| Outlier scan | IQR rule found a handful of unusually low reading scores; logged for possible capping.|



# **3. Data Transformation and Feature Engineering**
Once the data was explored, I moved on to feature engineering and data transformation. Here, I applied various techniques such as encoding categorical variables, imputing missing data, and scaling/normalizing features.

**3.1 Data Ingestion and Transformation**:

Data Ingestion: Automated the data extraction process to ensure that the data is always up-to-date.

Transformation Pipeline: Built a modular and efficient pipeline for preprocessing the data using transformation.py, ensuring the same transformations are applied to both training and test data.

# **4. Model Selection and Training**
In this stage, I focused on building and training various models. I experimented with different algorithms and hyperparameters to find the best model for the problem.

**4.1 Hyperparameter Tuning**:

Although hyperparameter optimization may seem like overkill for a small dataset, I included it for analysis purposes. The model_trainer.py file was set up to return the best model with the best evaluation metrics. Below are the models used and the hyperparameters tuned:

| **Model**                     | **Hyperparameters Tuned**                                                                                                                      |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Linear Regression**          | No parameters to tune here, it uses default settings.                                                                                           |
| **Logistic Regression**        | `C` (Regularization strength), `solver` (Optimization algorithm)                                                                                |
| **Random Forest Classifier**   | `n_estimators` (Number of trees), `max_depth` (Maximum depth of trees), `min_samples_split` (Minimum samples to split node), `min_samples_leaf` (Minimum samples at leaf) |
| **Support Vector Machine (SVM)** | `C` (Regularization parameter), `kernel` (Kernel type), `gamma` (Kernel coefficient)                                                           |
| **K-Nearest Neighbors (KNN)**  | `n_neighbors` (Number of neighbors), `weights` (Weight function used for prediction)                                                           |
| **Gradient Boosting Classifier** | `n_estimators` (Number of boosting stages), `learning_rate` (Learning rate), `max_depth` (Maximum depth of trees)                            |
| **Naive Bayes**                | `var_smoothing` (Portion of largest variance added for stability)                                                                               |
| **XGBRegressor**               | `learning_rate` [0.1, 0.01, 0.05, 0.001], `n_estimators` [8, 16, 32, 64, 128, 256]                                                            |
| **CatBoosting Regressor**      | `depth` [6, 8, 10], `learning_rate` [0.01, 0.05, 0.1], `iterations` [30, 50, 100]                                                               |
| **AdaBoost Regressor**         | `learning_rate` [0.1, 0.01, 0.5, 0.001], `n_estimators` [8, 16, 32, 64, 128, 256]                                                              |


**4.2 Model Results**:

Here are the results of the models based on the R2_Score:


| **Index** | **Model Name**            | **R2_Score** |
|-----------|---------------------------|--------------|
| 0         | Ridge                     | 0.880592     |
| 1         | Linear Regression          | 0.880433     |
| 2         | CatBoosting Regressor      | 0.851831     |
| 3         | AdaBoost Regressor         | 0.850582     |
| 4         | Random Forest Regressor    | 0.850013     |
| 5         | Lasso                      | 0.825447     |
| 6         | XGBRegressor               | 0.821221     |
| 7         | K-Neighbors Regressor      | 0.783958     |
| 8         | Decision Tree              | 0.726656     |


# **5. Monitoring, Logging, and Exception Handling**
To ensure the stability and smooth execution of the ML pipeline, I integrated efficient logging to capture important events at different severity levels such as info, debug, and error. I made sure to log significant actions such as:

Data loading, Model training, Model evaluation

Furthermore, I implemented exception handling throughout the code to maintain stability. I used custom exception classes to handle errors in a controlled manner and implemented try-catch blocks where suitable to prevent the pipeline from crashing.

Throughout the duration of the project, I also utilized Git for version control to manage and document the changes I made, ensuring a clean and organized development process.

# **6. Prediction Pipeline and Deployment**
Finally, I built a prediction pipeline to make real-time predictions using the trained model. This pipeline used:

**6.1. Flask Web Application**: 

Using Basic Flask application to handle web requests and render HTML templates. Creates an HTML form where users can enter their data. Upon form submission, the input data is processed, and a prediction is made using the trained machine learning model. The prediction results are displayed on the same page, providing immediate feedback to the user. This approach allows for an interactive user experience, making machine learning predictions accessible through a simple web interface.

A/N : The html UI does not look so great yet but it is a steep learning curve xD

![image](https://github.com/user-attachments/assets/d3571763-53e9-462e-a86c-5ca016444ff6)



**6.2: AWS Cloud: For deployment with CI/CD pipelines in the AWS Management Console.**:

Deployed the trained machine learning model to the AWS Management Console Cloud to make it available for real-time predictions. I set up Continuous Integration and Continuous Deployment (CI/CD) pipelines to automate the entire deployment process, which helped me seamlessly integrate and deploy updates directly from my Github repo to the model or application without manual intervention.

![image](https://github.com/user-attachments/assets/556b7a80-5690-412f-9947-faee4d7e95d8)



Author's note: Upcoming To do's:
1. Deploy with Azure cloud using github actions
2. Using Docker and AWS for conrtainerizing and deployment
   








