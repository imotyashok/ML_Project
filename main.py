import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
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

print(">>> Preprocessing data...")
print("Dataset shape: " + str(election_data.shape))

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

# Splitting data into 70% test, 15% validate, 15% train
print("\n>>> Splitting data into 70% train, 15% validate, and 15% test sets:")
train, validate, test = np.split(election_data.sample(frac=1), [int(.7*len(election_data)), int(.85*len(election_data))])

# Preprocessing train x and y
train = np.array(train).astype("float")
train_x = train[:, :-1]
train_y = train[:, -1]
print("Training set shape: " + str(train.shape))

# Preprocessing validation x and y
validate = np.array(validate).astype("float")
validate_x = validate[:, :-1]
validate_y = validate[:, -1]
print("Validate set shape: " + str(validate.shape))

# Preprocessing test x and y
test = np.array(test).astype("float")
test_x = test[:, :-1]
test_y = test[:, -1]
print("Test set shape: " + str(test.shape))

#-------------------------------------- BEGIN MODEL TESTING HERE ----------------------------------------------------

print("\n>>> Beginning model testing on validate data set...")
print("Model 1: Logistic Regression, C=1, penalty=l1")
lr = LogisticRegression(penalty='l1', C=1.0, random_state=1, solver='liblinear', multi_class='ovr')
lr.fit(train_x, train_y)
accuracy = lr.score(validate_x, validate_y)
print("Accuracy: %.3f" % accuracy)

print("\nModel 2: Logistic Regression, C=0.001, penalty=l1")
lr = LogisticRegression(penalty='l1', C=0.001, random_state=1, solver='liblinear', multi_class='ovr')
lr.fit(train_x, train_y)
accuracy = lr.score(validate_x, validate_y)
print("Accuracy: %.3f" % accuracy)

print("\nModel 3: Logistic Regression, C=100, penalty=l1")
lr = LogisticRegression(penalty='l1', C=100, random_state=1, solver='liblinear', multi_class='ovr')
lr.fit(train_x, train_y)
accuracy = lr.score(validate_x, validate_y)
print("Accuracy: %.3f" % accuracy)

print("\nModel 4: SVM, C=1.0")
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(train_x, train_y)
accuracy = svm.score(validate_x, validate_y)
print("Accuracy: %.3f" % accuracy)

print("\n>>> Beginning Baseline model training...")
print("Baseline Model 1: Strategy = \"stratified\"")
dummy = DummyClassifier(strategy="stratified")
dummy.fit(train_x, train_y)
dummy.predict(validate_x)
accuracy = dummy.score(validate_x, validate_y)
print("Accuracy : %.3f" % accuracy)

print("\nBaseline Model 2: Strategy = \"uniform\"")
dummy = DummyClassifier(strategy="uniform")
dummy.fit(train_x, train_y)
dummy.predict(validate_x)
accuracy = dummy.score(validate_x, validate_y)
print("Accuracy : %.3f" % accuracy)

print("\n>>> Model Analysis: Logistic Regression and SVM give very similar results; best model is most likely Model 1")

print("\n>>> Testing our best model on test data set:")
lr = LogisticRegression(penalty='l1', C=1.0, random_state=1, solver='liblinear', multi_class='ovr')
lr.fit(train_x, train_y)
accuracy = lr.score(test_x, test_y)
print("Accuracy: %.3f" % accuracy)

print("\nResults: Our model is around 90% accurate at predicting whether someone votes Democrat or Republican."+
      "\nThis model seems fairly reliable and well fitted, since the validation and test set accuracy are both around 90%."
      "\nOur baseline models' accuracies are around 50-75%, which means that our model performs significantly better than "
      "\na baseline model.")
