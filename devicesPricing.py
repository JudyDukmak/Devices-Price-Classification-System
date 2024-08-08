import pandas as pd 

# Load the train dataset
train_data = pd.read_csv('train - train.csv')

# Load the test dataset
test_data = pd.read_csv('test - test.csv')

print("Train Data Preview:")
# Preview the first few rows of the train dataset
print(train_data.head())

# Check for missing values
print(train_data.isnull().sum())

# Display basic information about the dataset
print(train_data.info())

# Display summary statistics
print(train_data.describe())

# Create a new feature for screen resolution
train_data['resolution'] = train_data['px_height'] * train_data['px_width']
test_data['resolution'] = test_data['px_height'] * test_data['px_width']
# remove px_height and px_width columns
train_data.drop(['px_height','px_width'],axis=1, inplace=True)
test_data.drop(['px_height', 'px_width'], axis=1, inplace=True)

# Check the first 5 rows to ensure the changes in the DataFrames
print(train_data.head())
print(test_data.head())

