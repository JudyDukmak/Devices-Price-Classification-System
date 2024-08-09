import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the train and test datasets
train_data = pd.read_csv('train - train.csv')
test_data = pd.read_csv('test - test.csv')

# Preview the first few rows of the train dataset
print("Train Data Preview:")
print(train_data.head())

# Check and handle missing values in the training dataset
print("Missing values in the training data:")
print(train_data.isnull().sum())
train_data.fillna(train_data.mean(), inplace=True)
print("Missing values after handling in the training data:")
print(train_data.isnull().sum())

# Check and handle missing values in the test dataset
print("Missing values in the test data:")
print(test_data.isnull().sum())
test_data.fillna(test_data.mean(), inplace=True)

# Feature Engineering
train_data['screen_resolution'] = train_data['px_height'] * train_data['px_width']
train_data['aspect_ratio'] = train_data['sc_h'] / train_data['sc_w']
train_data['battery_performance'] = train_data['battery_power'] / train_data['mobile_wt']

# Drop unnecessary columns
train_data.drop(['px_height', 'px_width', 'sc_h', 'sc_w', 'battery_power', 'mobile_wt'], axis=1, inplace=True)

# Apply the same feature engineering to the test dataset
test_data['screen_resolution'] = test_data['px_height'] * test_data['px_width']
test_data['aspect_ratio'] = test_data['sc_h'] / test_data['sc_w']
test_data['battery_performance'] = test_data['battery_power'] / test_data['mobile_wt']
test_data.drop(['px_height', 'px_width', 'sc_h', 'sc_w', 'battery_power', 'mobile_wt'], axis=1, inplace=True)

# Display updated data
print("Train data after feature engineering:")
print(train_data.head())
print("Test data after feature engineering:")
print(test_data.head())

# Replace infinte values with NaN in the aspect ratio column
train_data['aspect_ratio'] = train_data['aspect_ratio'].replace(np.inf, np.nan)
test_data['aspect_ratio'] = test_data['aspect_ratio'].replace(np.inf, np.nan)

# Fill NaN with the median value
train_data['aspect_ratio'] = train_data['aspect_ratio'].fillna(train_data['aspect_ratio'].median())
test_data['aspect_ratio'] = test_data['aspect_ratio'].fillna(test_data['aspect_ratio'].median())

# Display basic information and statistics
print("Training data info:")
print(train_data.info())
print("Training data statistics:")
print(train_data.describe())


# Correlation Matrix
plt.figure(figsize=(14, 10))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of Target Variable
sns.countplot(x='price_range', data=train_data)
plt.title('Distribution of Price Range')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ram', y='battery_performance', hue='price_range', data=train_data)
plt.title('RAM vs Battery Performance by Price Range')
plt.show()

# Feature Scaling
scaler = StandardScaler()
train_data[['ram', 'battery_performance']] = scaler.fit_transform(train_data[['ram', 'battery_performance']])
test_data[['ram', 'battery_performance']] = scaler.transform(test_data[['ram', 'battery_performance']])

# Split the data into features and target variable
X = train_data.drop('price_range', axis=1)
y = train_data['price_range']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle infinite and NaN values in training and validation sets
X_train = np.where(np.isinf(X_train), np.nan, X_train)
X_val = np.where(np.isinf(X_val), np.nan, X_val)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Verify that there are no NaNs or Infs left
print("NaNs in X_train after imputation:", np.isnan(X_train).sum())
print("Infs in X_train after imputation:", np.isinf(X_train).sum())
print("NaNs in X_val after imputation:", np.isnan(X_val).sum())
print("Infs in X_val after imputation:", np.isinf(X_val).sum())

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
con_mat = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(con_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_val, y_pred))
