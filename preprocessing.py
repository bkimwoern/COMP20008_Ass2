import numpy as np
import pandas as pd

def preprocessing():
    """Pre-processing person_csv"""
    person_csv = pd.read_csv('datasets/datasets-100/person-100.csv')
    process_person_csv(person_csv)

    accident_csv = pd.read_csv('datasets/datasets-100/accident-100.csv')

def filter_out_value(record, column, value):
    """ Filters a DataFrame based on a given column-value pair """
    return record[record[column] == value]

def process_person_csv(person_csv):
    filtered_person = person_csv

    # --- Imputing blanks in SEATING_POSITION ---
    # The only blanks in SEATING_POSITION are for 'drivers' in ROAD_USER_TYPE
    # Normalising blanks to nan values
    filtered_person['SEATING_POSITION'].replace('', np.nan, inplace=True)
    driver_mask = (filtered_person['SEATING_POSITION'].isna() &
                   (filtered_person['ROAD_USER_TYPE_DESC'] == 'Drivers'))
    # Imputing these rows to '1'
    filtered_person.loc[driver_mask, 'SEATING_POSITION'] = '1'

    # --- Creating new column indicating whether a person was in an enclosed vehicle ---
    # Defaulting all values to 0 (is not in enclosed vehicle)
    filtered_person['IN_METAL_BOX'] = 0
    filtered_person.loc[filtered_person['ROAD_USER_TYPE_DESC'].isin(['Drivers', 'Passengers']), 'IN_METAL_BOX'] = 1


    filtered_person.to_csv('datasets/filtered_person.csv', index=False)