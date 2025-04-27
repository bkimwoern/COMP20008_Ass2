import numpy as np
import pandas as pd

def preprocessing():
    """Pre-processing person_csv"""
    person_csv = pd.read_csv('datasets/datasets-100/person-100.csv')
    process_person_csv(person_csv)

    #accident_csv = pd.read_csv('datasets/datasets-100/accident-100.csv')
    accident_csv = pd.read_csv('datasets/accident.csv')
    process_accident_csv(accident_csv)
    #print(accident_csv[accident_csv['DAY_OF_WEEK'] == 1])

def filter_out_value(record, column, value):
    """ Filters a DataFrame based on a given column-value pair """
    return record[record[column] == value]

def process_person_csv(person_csv):
    filtered_person = person_csv

    # --- Imputing blanks in SEATING_POSITION ---
    # The only blanks in SEATING_POSITION are for 'drivers' in ROAD_USER_TYPE
    # Normalising blanks to nan values
    filtered_person.replace({'SEATING_POSITION': np.nan}, inplace=True)
    driver_mask = (filtered_person['SEATING_POSITION'].isna() &
                   (filtered_person['ROAD_USER_TYPE_DESC'] == 'Drivers'))
    # Imputing these rows to '1'
    filtered_person.loc[driver_mask, 'SEATING_POSITION'] = '1'

    # --- Creating new column indicating whether a person was in an enclosed vehicle ---
    # Defaulting all values to 0 (is not in enclosed vehicle)
    filtered_person['IN_METAL_BOX'] = 0
    filtered_person.loc[filtered_person['ROAD_USER_TYPE_DESC'].isin(['Drivers', 'Passengers']), 'IN_METAL_BOX'] = 1

    # --- May have to impute EJECTED_CODE values = 9 to 0.


    filtered_person.to_csv('datasets/filtered_person.csv', index=False)

def process_accident_csv(accident_csv):
    # --- Normalising date values in accident_csv ---
    accident_csv['ACCIDENT_DATE'] = pd.to_datetime(accident_csv['ACCIDENT_DATE'], format='%d/%m/%Y')

    # --- Creating a filtered accident csv ---
    filtered_accident = accident_csv

    # --- Adding a public holiday column to filtered_accident csv ---
    public_holiday_column(filtered_accident)
    night_day_column(filtered_accident)
    day_of_week(filtered_accident)

    filtered_accident.to_csv('datasets/filtered_accident.csv', index=False)

def public_holiday_column(filtered_accident):
    # --- Normalising values in public_holiday_csv
    public_holiday_csv = pd.read_csv('datasets/public_holiday_2012-2024.csv')
    public_holiday_csv['Date'] = pd.to_datetime(public_holiday_csv['Date'], format='%d/%m/%Y')
    public_holiday_csv['National_holiday'] = public_holiday_csv['National_holiday'].astype(bool)

    # Extracting dates from accident_csv that fall on national holidays
    national_holiday = public_holiday_csv.loc[public_holiday_csv['National_holiday'] == True, 'Date']

    # --- Creating a new column in filtered_accident called PUBLIC_HOLIDAY
    # Initialising everything to false
    filtered_accident['PUBLIC_HOLIDAY'] = 0
    # Dates that fall on public holidays are set to 1
    filtered_accident.loc[filtered_accident['ACCIDENT_DATE'].isin(national_holiday), 'PUBLIC_HOLIDAY'] = 1

    # Possibly include regional holidays???

def night_day_column(filtered_accident):
    # --- Creating new column 'DAY'- 0 if night, 1 if day ---
    # Light condition 1 = day, 2 = dusk/dawn
    filtered_accident['DAY'] = 0
    filtered_accident.loc[filtered_accident['LIGHT_CONDITION'].isin([1,2]),'DAY'] = 1

def day_of_week(filtered_accident):
    # --- Identifying mismatches in accident_csv ---
    # Computing the expected DAY_OF_WEEK values from the description
    expected = filtered_accident['DAY_WEEK_DESC'].map(day_of_week_map)

    # Helper print statements to see incorrect rows
    #mismatched = filtered_accident[filtered_accident['DAY_OF_WEEK'] != expected]
    #print("Bad rows:\n", mismatched[['ACCIDENT_NO', 'DAY_OF_WEEK', 'DAY_WEEK_DESC']])

    # Correcting only the mismatched rows within the dataset
    filtered_accident.loc[filtered_accident['DAY_OF_WEEK'] != expected, 'DAY_OF_WEEK'] = expected

day_of_week_map = {
    'Sunday': 1,
    'Monday': 2,
    'Tuesday': 3,
    'Wednesday': 4,
    'Thursday': 5,
    'Friday': 6,
    'Saturday': 7,
}