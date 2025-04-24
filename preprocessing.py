import pandas as pd

def preprocessing():
    """Pre-processing person_csv"""
    person_csv = pd.read_csv('datasets/datasets-100/person-100.csv')

    # replacing the previous age column with new one containing broader ranges
    # dropping 'Unknown' age group, as only makes up 3% of dataset.
    age_group = person_csv['AGE_GROUP'].map(categorise_age)
    person_csv['AGE_GROUP'] = age_group

    print(age_group)

    fatalities_csv = filter_out_value(person_csv, 'INJ_LEVEL', 1)
    serious_injury_csv = filter_out_value(person_csv, 'INJ_LEVEL', 2)
    other_injury_csv = filter_out_value(person_csv, 'INJ_LEVEL', 3)
    not_injured_csv = filter_out_value(person_csv, 'INJ_LEVEL', 4)

    #print(fatalities_csv.head(100))
    #print(serious_injury_csv.head(100))
    #print(other_injury_csv.head(100))
    #print(not_injured_csv.head(100))

""" From task 1_1, assignment 1"""
def categorise_age(age_group):
    """ Categorises age groups into broader ranges"""
    if age_group in ['13-15']:
        return 'Under 16'
    elif age_group in ['16-17', '18-21', '22-25']:
        return '16-25'
    elif age_group in ['26-29', '30-39']:
        return '26-39'
    elif age_group in ['40-49', '50-59', '60-64']:
        return '40-64'
    elif age_group in ['65-69', '70+']:
        return '65+'

def filter_out_value(record, column, value):
    """ Filters a DataFrame based on a given column-value pair """
    return record[record[column] == value]