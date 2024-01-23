# Data-Science-and-AI-Group-5

# Import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

# Read the dataset into a DataFrame
df = pd.read_csv('C:/Users/EFREN/Downloads/ph_dengue_cases2016-2020.csv')

# Display the first few rows of the dataset
df.head()

# Check for missing values
nan_count = df.isna().sum()

print(nan_count)

# Handle missing values (if any)
df.fillna(0)

# For simplicity, you can drop rows with missing values
df.loc[~df.isnull().any(axis=1)]

# Check for duplicate rows
df.duplicated()

#Group regions into Island groups
luzon_regions = ('Region I', 'Region II', 'Region III', 'Region IV-A', 'Region IV-B', 'Region V', 'CAR', 'NCR')
visayas_regions =('Region VI', 'Region VII', 'Region VIII')
mindanao_regions = ('Region IX', 'Region X', 'Region XI', 'Region XII', 'Region XIII', 'BARMM')

#Assign numbers relative to the months
month_name_to_num = {name: num for num, name in enumerate(df['Month'].unique(), start=1)}
month_num_to_abb = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

#Create columns using panda
df['month_num'] = df['Month'].map(month_name_to_num)
df['month_abb'] = df['month_num'].map(month_num_to_abb)
df['year_month'] = df['Year'].astype(str) + '-' + df['month_num'].astype(str)

# Create an 'island' column using the provided conditions
df['island'] = np.select([df['Region'].isin(luzon_regions),
                           df['Region'].isin(visayas_regions)],
                          ['Luzon', 'Visayas'],
                          default='Mindanao')

                          #Convert 'month' to a categorical value
df['month_abb'] = pd.Categorical(df['month_abb'], categories=month_num_to_abb.values(), ordered=True)

#Convert 'year' to a categorical value
df['Year'] = pd.Categorical(df['Year'], ordered=True)

#Extract numeric from 'year_month'
df['year_month'] = df['year_month'].str[2:]

#Convert 'year_month' to a categorical value
df['year_month'] = pd.Categorical(df['year_month'], ordered=True)

#Display first 10 rows
df.head(10)

#Install packages
import plotly.express as px
!pip install plotly

# Group by month_abb and Year and calculate the sum of Dengue_Cases
seasonal_trend = df.groupby(['month_abb', 'Year']).agg({'Dengue_Cases': 'sum'}).reset_index()

# Group by month_abb and Year and calculate the sum of Dengue_Cases
seasonal_trend = df.groupby(['month_abb', 'Year']).agg({'Dengue_Cases': 'sum'}).reset_index()
​
fig = px.line(seasonal_trend, x='month_abb', y='Dengue_Cases', color='Year',
              title='Seasonal Trend of Dengue Cases per Year',
              labels={'month_abb': 'Month', 'Dengue_Cases': 'Dengue Cases'},
              line_shape='linear', render_mode='svg')
# Adjust layout
fig.update_layout(
    xaxis=dict(tickangle=45),
    yaxis=dict(title='Dengue Cases'),
    legend_title='Year'
)
# Group by Region and calculate the mean of Dengue_Cases
average_cases = df.groupby('Region')['Dengue_Cases'].mean().reset_index()
​
# Sort by Avg_cases in descending order
average_cases = average_cases.sort_values(by='Dengue_Cases', ascending=False)
# Plot the average number of Dengue cases per region using Plotly bar chart
fig = px.bar(average_cases, y='Region', x='Dengue_Cases', orientation='h',
             title='Average Number of Dengue Cases per Region',
             labels={'Dengue_Cases': 'Average No. of Cases', 'Region': 'Region'},
             color='Dengue_Cases', color_continuous_scale='reds',
             text='Dengue_Cases', height=700)
#Show the plot
fig.show()
plt.figure(figsize=(5, 5))
sns.lineplot(x='Year', y='Dengue_Cases', hue='Region', data=df, legend=False)
plt.show()
Step 2: Linear Regression

Implement a simple linear regression model to predict a continuous variable

#isolate Region IV-A
region_data = df[df['Region'] == 'Region IV-A']
plt.figure(figsize=(10,10))
​
plt.scatter(region_data['Dengue_Cases'], region_data['Dengue_Deaths'])
plt.title('Dengue case-to-death ratio in Region IV-A')
plt.xlabel('Dengue_Cases')
plt.ylabel('Dengue_Deaths')
plt.show()
#isolate Region IV-A
region_data = df[df['Region'] == 'Region IV-A']
plt.figure(figsize=(10,10))
​
​
plt.scatter(region_data['Year'], region_data['Dengue_Cases'])
plt.title('Dengue transmission rate through the years in Region IV-A')
plt.xlabel('Year')
plt.ylabel('Dengue_Cases')
plt.show()
#isolate Region IV-A
region_data = df[df['Region'] == 'Region IV-A']
plt.figure(figsize=(10,10))
​
​
plt.scatter(region_data['Year'], region_data['Dengue_Deaths'])
plt.title('Fatality Rate in Region IV-A')
plt.xlabel('Year')
plt.ylabel('Dengue_Deaths')
plt.show()
​
# First dataset
X1 = df[['Dengue_Cases']]
y1 = df['Dengue_Deaths']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
​
model1 = LinearRegression()
model1.fit(X1_train, y1_train)
​
# Second dataset
X2 = df[['Year']] 
y2 = df['Dengue_Cases']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
​
model2 = LinearRegression()
model2.fit(X2_train, y2_train)
​
# Third dataset
X3 = df[['Year']]  
y3 = df['Dengue_Deaths']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
​
model3 = LinearRegression()
model3.fit(X3_train, y3_train)
​
​
​
# Predictions for the first model
y1_pred = model1.predict(X1_test)
​
# Predictions for the second model
y2_pred = model2.predict(X2_test)
​
# Predictions for the third model
y3_pred = model3.predict(X3_test)
Evaluate the model using appropriate metrics (e.g., Mean Squared Error)

mae1 = mean_absolute_error(y1_test, y1_pred)
mse1 = mean_squared_error(y1_test, y1_pred)
print(f'MAE: {mae1}')
print(f'MSE: {mse1}')
mae2 = mean_absolute_error(y2_test, y2_pred)
mse2 = mean_squared_error(y2_test, y2_pred)
print(f'MAE: {mae2}')
print(f'MSE: {mse2}')
mae3 = mean_absolute_error(y3_test, y3_pred)
mse3 = mean_squared_error(y3_test, y3_pred)
print(f'MAE: {mae3}')
print(f'MSE: {mse3}')
Visualize the relationship between the independent and dependent variables

plt.scatter(region_data['Dengue_Cases'], region_data['Dengue_Deaths'])
plt.plot(X1_test, y1_pred, color='red', linewidth=2)
plt.title('Dengue case-to-death ratio in Region IV-A')
plt.xlabel('Dengue_Cases')
plt.ylabel('Dengue_Deaths')
plt.show()
# Assuming 'region_data' is your DataFrame containing the data
filtered_data = region_data[(region_data['Year'] >= 2016) & (region_data['Year'] <= 2020)]
​
plt.scatter(filtered_data['Year'], filtered_data['Dengue_Cases'])
plt.plot(filtered_data['Year'], y1_pred[:len(filtered_data)], color='red', linewidth=2)
plt.title('Dengue transmission rate from 2016 to 2020 in Region IV-A')
plt.xlabel('Year')
plt.ylabel('Dengue Cases')
​
# Set x-axis ticks without 0.5
plt.xticks(range(2016, 2021))
​
plt.show()
#isolate Region IV-A
region_data = df[df['Region'] == 'Region IV-A']
plt.figure(figsize=(10,10))
​
​
plt.scatter(region_data['Year'], region_data['Dengue_Deaths'])
plt.plot(filtered_data['Year'], y1_pred[:len(filtered_data)], color='red', linewidth=2)
plt.title('Fatality Rate in Region IV-A')
plt.xlabel('Year')
plt.ylabel('Dengue_Deaths')
plt.show()
​
​
Step 3: Logistic Regression

Split the dataset into training and testing sets

# Set a threshold to classify as high or low production
threshold = 50000  # You can adjust this threshold as per your requirement
​
# For the first dataset (changing names)
df['Dengue_Cases'] = (df['Dengue_Deaths'] > threshold).astype(int)
X_ration = df[['Dengue_Cases']]
y_ration = df['Dengue_Deaths']
X_ration_train, X_ration_test, y_ration_train, y_ration_test = train_test_split(X_ration, y_ration, test_size=0.2, random_state=42)
​
# For the second dataset (changing names)
df['Year'] = (df['Dengue_Cases'] > threshold).astype(int)
X_case = df[['Year']]
y_case = df['Dengue_Cases']
X_case_train, X_case_test, y_case_train, y_case_test = train_test_split(X_case, y_case, test_size=0.2, random_state=42)
​
# For the third dataset (changing names)
df['Year'] = (df['Dengue_Deaths'] > threshold).astype(int)
X_fatality = df[['Year']]
y_fatality = df['Dengue_Deaths']
X_fatality_train, X_fatality_test, y_fatality_train, y_fatality_test = train_test_split(X_fatality, y_fatality, test_size=0.2, random_state=42)
Train the logistic regression model and evaluate its performance using accuracy, precision, recall, and F1-score

# Create and train the first model
model1 = LogisticRegression()
model1.fit(X1_train, y1_train)
​
# Create and train the second model
model2 = LogisticRegression()
model2.fit(X2_train, y2_train)
​
# Create and train the third model
model3 = LogisticRegression()
model3.fit(X3_train, y3_train)
​
# Make predictions for the first model
y1_pred = model1.predict(X1_test)
​
# Make predictions for the second model
y2_pred = model2.predict(X2_test)
​
# Make predictions for the third model
y3_pred = model3.predict(X3_test)
​
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
​
# Calculate metrics for each model
accuracy1 = accuracy_score(y1_test, y1_pred)
precision1 = precision_score(y1_test, y1_pred, average='weighted')  # Use 'weighted' for multiclass
recall1 = recall_score(y1_test, y1_pred, average='weighted')  # Use 'weighted' for multiclass
f1_1 = f1_score(y1_test, y1_pred, average='weighted')  # Use 'weighted' for multiclass
​
# Print the metrics
print(f"Accuracy: {accuracy1}")
print(f"Precision: {precision1}")
print(f"Recall: {recall1}")
print(f"F1 Score: {f1_1}")
# New data point for prediction
new_data_point = np.array([[10]])
​
# Make prediction using the appropriate logistic regression model (replace model1 with the actual model name)
prediction = model1.predict(new_data_point)
​
# Interpret the prediction
if prediction[0] == 1:
    result = "High Transmission"
else:
    result = "Low Transmission"
​
print(f"Prediction for the given feature value (10): {result}")
​
# Assuming you have a trained logistic regression model named 'model1' for Dengue_Deaths
# You need to replace 'model1' with the actual name of your trained model
​
# Generate x values for the plot
x_values = np.linspace(X1_train.min(), X1_train.max(), 100).reshape(-1, 1)
​
# Adjust the coefficients to shift the curve
adjusted_intercept = model1.intercept_[0] - np.log(1/model1.coef_[0] - 1)
​
# Calculate corresponding y values using the logistic function
y_values = 1 / (1 + np.exp(-(model1.coef_[0] * x_values + adjusted_intercept)))
​
# Scatter plot of data points
plt.scatter(X1_train, y1_train, color='blue', label='Data Points')
​
# Plot the adjusted logistic regression curve
plt.plot(x_values, y_values, color='red', label='Adjusted Logistic Regression Curve')
​
plt.xlabel('Dengue_Cases')
plt.ylabel('Dengue_Deaths Probability (1) / (0)')  # Adjust label based on your classification labels
plt.legend()
plt.show()
​
Step 4: Classification Models

Explore other classification models such as Decision Trees, Random Forest, Support Vector Machines (SVM), etc.

Decision Trees

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
​
# Assuming you have features X1_train and labels y1_train from linear regression
# Replace with your actual variable names if different
​
# Create a decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
​
# Train the model
dt_classifier.fit(X1_train, y1_train)
​
# Make predictions
y_pred_dt = dt_classifier.predict(X1_test)
​
# Evaluate the model
accuracy_dt = accuracy_score(y1_test, y_pred_dt)
classification_report_dt = classification_report(y1_test, y_pred_dt)
​
print(f"Decision Tree Accuracy: {accuracy_dt}")
print("Classification Report for Decision Tree:")
print(classification_report_dt)
​
Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
​
# Create a random forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
​
# Train the model
rf_classifier.fit(X1_train, y1_train)
​
# Make predictions
y_pred_rf = rf_classifier.predict(X1_test)
​
# Evaluate the logistic regression model
accuracy_rf = accuracy_score(y1_test, y_pred_rf)
classification_report_rf = classification_report(y1_test, y_pred_rf)
​
print(f"Random Forest Accuracy: {accuracy_rf}")
print("Classification Report for Random Forest:")
print(classification_report_rf)
​
​
SVC

from sklearn.svm import SVC
​
# Create an SVM classifier
svm_classifier = SVC(random_state=42)
​
# Train the model
svm_classifier.fit(X1_train, y1_train)
​
# Make predictions
y_pred_svm = svm_classifier.predict(X1_test)
​
# Evaluate the model
accuracy_svm = accuracy_score(y1_test, y_pred_svm)
classification_report_svm = classification_report(y1_test, y_pred_svm)
​
print(f"SVM Accuracy: {accuracy_svm}")
print("Classification Report for SVM:")
print(classification_report_svm)
​
Compare the performance of these models using appropriate metrics

# Assuming you have the true labels (y1_test) and predictions from different models
​
# Decision Tree Metrics
accuracy_dt = accuracy_score(y1_test, y_pred_dt)
precision_dt = precision_score(y1_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y1_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y1_test, y_pred_dt, average='weighted')
​
print(f"\nMetrics for Decision Tree:\n")
print(f"Accuracy: {accuracy_dt}")
print(f"Precision: {precision_dt}")
print(f"Recall: {recall_dt}")
print(f"F1 Score: {f1_dt}")
​
# Random Forest Metrics
accuracy_rf = accuracy_score(y1_test, y_pred_rf)
precision_rf = precision_score(y1_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y1_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y1_test, y_pred_rf, average='weighted')
​
print(f"\nMetrics for Random Forest:\n")
print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1 Score: {f1_rf}")
​
# SVM Metrics
accuracy_svm = accuracy_score(y1_test, y_pred_svm)
precision_svm = precision_score(y1_test, y_pred_svm, average='weighted')
recall_svm = recall_score(y1_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y1_test, y_pred_svm, average='weighted')
​
print(f"\nMetrics for SVM:\n")
print(f"Accuracy: {accuracy_svm}")
print(f"Precision: {precision_svm}")
print(f"Recall: {recall_svm}")
print(f"F1 Score: {f1_svm}")
​
Visualize the results and the decision boundaries if applicable

!pip install mlxtend
from mlxtend.plotting import plot_decision_regions
​
# Assuming you have a trained model named 'model' (replace it with your actual model name)
model = dt_classifier  # Change the model based on your requirement
​
# Plotting decision boundary
plt.figure(figsize=(10, 8))
plot_decision_regions(X1_test.values, y1_test.values, clf=model, legend=2)
​
# Set labels and title
plt.xlabel(X1_test.columns[0])
plt.ylabel(y1_test.name)
plt.title('Decision Boundary')
​
# Show the plot
plt.show()
​
Step 5: Model Comparison and Conclusion

Compare the performance of all models used in the project

# Compare the performance of all models
print(f"Linear Regression MAE: {mae1}")
print(f"Logistic Regression Accuracy: {accuracy1}")
print(f"Decision Tree Accuracy: {accuracy_dt}")
print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"SVM Accuracy: {accuracy_svm}")
​
