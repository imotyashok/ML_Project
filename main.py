import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

election_data = pd.read_csv('usa-2016-presidential-election-by-county.csv', delimiter=";")
# print(election_data.shape)

# For testing purposes
#print(election_data.columns.values.tolist())
#pd.set_option('display.max_columns', None)

# This dataset has 159 columns and 3143 rows; we don't need most of the columns. We're just interested in:
#   'State', 'ST', 'County', 'Republicans 2016', 'Democrats 2016', 'Less Than High School Diploma', 'At Least High School Diploma',
#   "At Least Bachelors's Degree", 'Graduate Degree', 'White (Not Latino) Population', 'African American Population',
#   'Native American Population', 'Asian American Population', 'Other Race or Races', 'Latino Population', 'Total Population',
# We'll keep ST and County for now in case we'll need it later
election_data = election_data[['ST', 'County', 'Republicans 2016', 'Democrats 2016', 'Less Than High School Diploma',
                'At Least High School Diploma', "At Least Bachelors's Degree", 'Graduate Degree', 'White (Not Latino) Population',
                'African American Population', 'Native American Population', 'Asian American Population', 'Other Race or Races',
                'Latino Population']]

# Getting rid of any null values
election_data = election_data.dropna()

# Now we will prepare our labels: we'll use -1 for Republican, and 1 for Democrat;
# If Republican value is > 50, then set label = -1,
# If Democrat value is > 50, then set label = 1
#
print(election_data.shape)

data_labels = []
for value in election_data['Republicans 2016']:
    if value > 50.0:
        data_labels.append(-1)
    elif value <= 50.0:
        data_labels.append(1)

# print(len(data_labels))

# Testing to make sure our labels aligned correctly
# for i in range(100):
#     print("%f -- %d" % (election_data.iloc[i, 2], data_labels[i] ))
# Seems pretty good!

# Getting rid of any columns that don't have numerical data
election_data = election_data.drop(columns=['ST', 'County', 'Republicans 2016', 'Democrats 2016'])

# Adding the labels column into the dataset
election_data["Labels"] = data_labels
print(election_data.columns)

# Splitting data into 70% test, 15% validate, 15% train
train, validate, test = np.split(election_data.sample(frac=1), [int(.7*len(election_data)), int(.85*len(election_data))])

# Preprocessing train x and y
train = np.array(train).astype("float")
train_x = train[:, :-1]
train_y = train[:, -1]
print(train.shape)

# Preprocessing validation x and y
validate = np.array(validate).astype("float")
validate_x = validate[:, :-1]
validate_y = validate[:, -1]
print(validate.shape)

# Preprocessing test x and y
test = np.array(test).astype("float")
test_x = test[:, :-1]
test_y = test[:, -1]
print(test.shape)

#-------------------------------------- BEGIN MODEL TESTING HERE ----------------------------------------------------

print("\n>>> Beginning model testing...")
print("Model 1: Logistic Regression, C=1, penalty=l1")
lr = LogisticRegression(penalty='l1', C=1.0, random_state=1, solver='liblinear', multi_class='ovr')
lr.fit(train_x, train_y)
accuracy = lr.score(validate_x, validate_y)
print("Accuracy: %.3f" % accuracy)





