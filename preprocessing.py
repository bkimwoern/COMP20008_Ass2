import numpy as np
import pandas as pd

def preprocessing():
    """Pre-processing person_csv"""
    person_csv = pd.read_csv('datasets/person.csv')
    process_person_csv(person_csv)

    #accident_csv = pd.read_csv('datasets/datasets-100/accident-100.csv')
    accident_csv = pd.read_csv('datasets/accident.csv')
    process_accident_csv(accident_csv)

    filtered_vehicle_csv = pd.read_csv('datasets/filtered_vehicle.csv')
    process_vehicle_body_type(filtered_vehicle_csv)
    filtered_vehicle_csv.to_csv('datasets/filtered_vehicle_new.csv', index=False)


def filter_out_value(record, column, value):
    """ Filters a DataFrame based on a given column-value pair """
    return record[record[column] == value]

def process_person_csv(person_csv):
    filtered_person = person_csv

    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_colwidth', None)

    #pd.reset_option('display.max_rows')

    # --- Imputing blanks in SEATING_POSITION ---
    #    The only blanks in SEATING_POSITION are for 'drivers' in ROAD_USER_TYPE
    #    Replacing blanks and 'nan's' with 'D'
    filtered_person.replace(r'^\s*$', 'D', regex=True, inplace=True)
    #drivers = filter_out_value(filtered_person, 'ROAD_USER_TYPE_DESC', 'Drivers')
    #print(drivers[['ACCIDENT_NO', 'SEATING_POSITION', 'ROAD_USER_TYPE_DESC']])

    # Converting unknown variations in dataset to 'nan'
    filtered_person.replace(['Unknown', 'Not Known', 'N/A', 'NK', '5-Dec', '0-4'], np.nan, inplace=True)

    # --- Creating new column indicating whether a person was in an enclosed vehicle ---
    #   Defaulting all values to 0 (is not in enclosed vehicle)
    filtered_person['IN_METAL_BOX'] = 0
    filtered_person.loc[filtered_person['ROAD_USER_TYPE_DESC'].isin(['Drivers', 'Passengers']), 'IN_METAL_BOX'] = 1
    filtered_person.loc[filtered_person['ROAD_USER_TYPE_DESC'].isna(), 'IN_METAL_BOX'] = np.nan

    # Imputing not known values in 'HELMET_BELT_WORN' column
    imputing_safety_equipment(filtered_person)

    filtered_person.to_csv('datasets/filtered_person.csv', index=False)

def process_accident_csv(accident_csv):
    # --- Normalising date values in accident_csv ---
    accident_csv['ACCIDENT_DATE'] = pd.to_datetime(accident_csv['ACCIDENT_DATE'], format='%d/%m/%Y')

    # --- Creating a filtered accident csv ---
    filtered_accident = accident_csv

    # --- Preprocessing accident_csv ---
    #   Adding a public holiday boolean column to filtered_accident csv
    public_holiday_column(filtered_accident)
    #   Adding a night or day column to filtered_accident csv
    night_day_column(filtered_accident)
    #   Fixing incorrect DAY_OF_WEEK values in filtered_accident csv
    day_of_week(filtered_accident)
    # Adding a column that indicates whether accident occurred at an intersection (1) or not (0)
    at_intersection(filtered_accident)

    filtered_accident.to_csv('datasets/filtered_accident.csv', index=False)

def imputing_safety_equipment(filtered_person):

    # --- Creating a new column 'UNPROTECTED'- 1 if no safety equipment was worn, 0 if worn ---
    filtered_person['UNPROTECTED'] = filtered_person['HELMET_BELT_WORN'].apply(
        lambda x: 0 if x in [1, 3, 6] # Wore safety equipment
        else 1 if x in [2, 4, 5, 7, 8] # Did not wear safety equipment
        else 2 # Unknown
    )

    # --- Extracting records of unknown safety equipment usage IN an encased vehicle ---
    unknown_encased = filter_out_value(filter_out_value(filtered_person, 'HELMET_BELT_WORN', 9.0),
                                       'IN_METAL_BOX', 1)
    print(unknown_encased)

    safety_worn_encased = filter_out_value(filter_out_value(filtered_person, 'UNPROTECTED', 0),
                                           'IN_METAL_BOX', 1 )
    print(safety_worn_encased['HELMET_BELT_WORN'].value_counts())
    #print(filter_out_value(safety_worn_encased, 'HELMET_BELT_WORN', 6.0))

    safety_not_worn_encased = filter_out_value(filter_out_value(filtered_person, 'UNPROTECTED', 1),
                                               'IN_METAL_BOX', 1)
    print(safety_not_worn_encased['HELMET_BELT_WORN'].value_counts())

    # --- Extracting records of unknown safety equipment usage NOT in an encased vehicle ---
    unknown_exposed = filter_out_value(filter_out_value(filtered_person, 'HELMET_BELT_WORN', 9.0),
                                       'IN_METAL_BOX', 0)
    print(unknown_exposed)

    safety_worn_exposed = filter_out_value(filter_out_value(filtered_person, 'UNPROTECTED', 0),
                                           'IN_METAL_BOX', 0)
    print(safety_worn_exposed['HELMET_BELT_WORN'].value_counts())

    safety_not_worn_exposed = filter_out_value(filter_out_value(filtered_person, 'UNPROTECTED', 1),
                                               'IN_METAL_BOX', 0)
    print(safety_not_worn_exposed['HELMET_BELT_WORN'].value_counts())


def public_holiday_column(filtered_accident):
    # --- Normalising values in public_holiday_csv
    public_holiday_csv = pd.read_csv('datasets/public_holiday_2012-2024.csv')
    public_holiday_csv['Date'] = pd.to_datetime(public_holiday_csv['Date'], format='%d/%m/%Y')
    public_holiday_csv['National_holiday'] = public_holiday_csv['National_holiday'].astype(bool)

    # --- Extracting dates from accident_csv that fall on national holidays ---
    national_holiday = public_holiday_csv.loc[public_holiday_csv['National_holiday'] == True, 'Date']

    # --- Creating a new column in filtered_accident called PUBLIC_HOLIDAY
    #   Initialising everything to false
    filtered_accident['PUBLIC_HOLIDAY'] = 0
    #   Dates that fall on national holidays are set to 1
    filtered_accident.loc[filtered_accident['ACCIDENT_DATE'].isin(national_holiday), 'PUBLIC_HOLIDAY'] = 1

    # Possibly include regional holidays???

def night_day_column(filtered_accident):
    # --- Creating new column 'DAY'- 0 if night, 1 if day ---
    # Light condition 1 = day, 2 = dusk/dawn
    filtered_accident['DAY'] = 0
    filtered_accident.loc[filtered_accident['LIGHT_CONDITION'].isin([1,2]),'DAY'] = 1

def day_of_week(filtered_accident):
    # --- Mapping for DAY_WEEK_DESC with matching DAY_OF_WEEK
    day_of_week_map = {
        'Sunday': 1,
        'Monday': 2,
        'Tuesday': 3,
        'Wednesday': 4,
        'Thursday': 5,
        'Friday': 6,
        'Saturday': 7,
    }

    # --- Identifying mismatches in accident_csv ---
    #   Computing the expected DAY_OF_WEEK values from the description
    expected = filtered_accident['DAY_WEEK_DESC'].map(day_of_week_map)

    #   Helper print statements to see incorrect rows
    #mismatched = filtered_accident[filtered_accident['DAY_OF_WEEK'] != expected]
    #print("Bad rows:\n", mismatched[['ACCIDENT_NO', 'DAY_OF_WEEK', 'DAY_WEEK_DESC']])

    #   Correcting only the mismatched rows within the dataset
    filtered_accident.loc[filtered_accident['DAY_OF_WEEK'] != expected, 'DAY_OF_WEEK'] = expected

def at_intersection(filtered_accident):
    filtered_accident['AT_INTERSECTION'] = filtered_accident['ROAD_GEOMETRY'].apply(lambda x: 1 if x in [1,2,3,4] else 0)


def process_vehicle_body_type(filtered_vehicle):
    vehicle_body_map = {
        'OF SHD': 'OF/SHD',
        'SEDAN' : 'SED',
    }
    filtered_vehicle['VEHICLE_BODY_STYLE'] = filtered_vehicle['VEHICLE_BODY_STYLE'].replace(vehicle_body_map)


""""
protection worn / (protection worn + protection not worn)

then select 0 or 1, where the chance of selecting 1 is the number above 


"""