import pandas as pd
import numpy as np

election_data = pd.read_csv('usa-2016-presidential-election-by-county.csv', delimiter=";")
# print(election_data.shape)

# For testing purposes
#print(election_data.columns.values.tolist())
#pd.set_option('display.max_columns', None)

# This dataset has 159 columns and 3143 rows; we don't need most of the columns. We're just interested in:
#   'State', 'ST', 'County', 'Republicans 2016', 'Democrats 2016', 'Less Than High School Diploma', 'At Least High School Diploma',
#   "At Least Bachelors's Degree", 'Graduate Degree', 'White (Not Latino) Population', 'African American Population',
#   'Native American Population', 'Asian American Population', 'Other Race or Races', 'Latino Population', 'Total Population',

election_data = election_data[['State', 'ST', 'County', 'Republicans 2016', 'Democrats 2016', 'Less Than High School Diploma',
                'At Least High School Diploma', "At Least Bachelors's Degree", 'Graduate Degree', 'White (Not Latino) Population',
                'African American Population', 'Native American Population', 'Asian American Population', 'Other Race or Races',
                'Latino Population', 'Total Population']]

# Getting rid of any null values
election_data = election_data.dropna()

# Now we will prepare our labels: we'll use -1 for Republican, and 1 for Democrat;
# If Republican value is > 50, then set label = -1,
# If Democrat value is > 50, then set label = 1
# If a value is == 50.... what do? ignore it?
print(election_data.shape)

data_labels = []
for value in election_data['Republicans 2016']:
    if value > 50.0:
        data_labels.append(-1)
    elif value <= 50.0:
        data_labels.append(1)

print(len(data_labels))

# Testing to make sure our labels aligned correctly
# for i in range(100):
#     print("%f -- %d" %(election_data.iloc[i, 3], data_labels[i] ))

election_data["Labels"] = data_labels





