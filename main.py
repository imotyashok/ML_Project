import pandas as pd

election_data = pd.read_csv('usa-2016-presidential-election-by-county.csv', delimiter=";")
print(election_data.shape)
#print(election_data.columns.values.tolist())

# This dataset has 159 columns and 3143 rows; we don't need most of the columns. We're just interested in:
#   'State', 'ST', 'County', 'Republicans 2016', 'Democrats 2016', 'Less Than High School Diploma', 'At Least High School Diploma',
#   "At Least Bachelors's Degree", 'Graduate Degree', 'White (Not Latino) Population', 'African American Population',
#   'Native American Population', 'Asian American Population', 'Other Race or Races', 'Latino Population', 'Total Population',

election_data = election_data[['State', 'ST', 'County', 'Republicans 2016', 'Democrats 2016', 'Less Than High School Diploma',
                'At Least High School Diploma', "At Least Bachelors's Degree", 'Graduate Degree', 'White (Not Latino) Population',
                'African American Population', 'Native American Population', 'Asian American Population', 'Other Race or Races',
                'Latino Population', 'Total Population']]

print(election_data.shape)