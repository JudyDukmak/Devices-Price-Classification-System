import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Load the train dataset
train_data = pd.read_csv('train - train.csv')

# Load the test dataset
test_data = pd.read_csv('test - test.csv')

print("Train Data Preview:")
# Preview the first few rows of the train dataset
print(train_data.head())

# Check for missing values in the training dataset
print(train_data.isnull().sum())
#fill missing values with the mean of each column
train_data.fillna(train_data.mean(), inplace=True)
#Verify Missing Values Are Handled
print("checking the missing values in the training data:")
print(train_data.isnull().sum())

# Check for missing values in the test dataset
print("checking the missing values in the test data:")
print(test_data.isnull().sum())

# Create a new feature for screen resolution  
train_data['screen_resolution'] = train_data['px_height'] * train_data['px_width']
# Create a new feature for aspect ratio
train_data['aspect_ratio'] = train_data['sc_h'] / train_data['sc_w']
# Create a new feature for battery performance
train_data['battery_performance'] = train_data['battery_power'] / train_data['mobile_wt']
# remove  columns from the training dataset
train_data.drop(['px_height','px_width'],axis=1, inplace=True)
train_data.drop(['sc_h', 'sc_w'], axis=1, inplace=True)
train_data.drop(['battery_power','mobile_wt'], axis=1, inplace=True)

# Apply the same feature to the test dataset
test_data['screen_resolution'] = test_data['px_height'] * test_data['px_width']
test_data['aspect_ratio'] = test_data['sc_h'] / test_data['sc_w']
test_data['battery_performance'] = test_data['battery_power'] / test_data['mobile_wt']
# remove  columns from the test dataset
test_data.drop(['px_height', 'px_width'], axis=1, inplace=True)
test_data.drop(['sc_h', 'sc_w'], axis=1, inplace=True)
test_data.drop(['battery_power','mobile_wt'], axis=1, inplace=True)

# Display the new features
print("preview train data after changing :")
print(train_data.head())
print("preview test data after changing :")
print(test_data.head())

# Display basic information about the training dataset
print(train_data.info())

# Display summary statistics
print(train_data.describe())

corr_matrix = train_data.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Split the data into features and target variable
X = train_df.drop('price_range', axis=1)
y = train_df['price_range']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_clf.fit(X_train, y_train)

# Predict on the validation set
y_pred = rf_clf.predict(X_val)

# Evaluate the model
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))
