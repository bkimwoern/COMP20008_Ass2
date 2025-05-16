import numpy as np
import pandas as pd
from .utils import filter_out_value

# --- Main function to process the person.csv dataset ---
def process_person_csv():
    filtered_person = pd.read_csv('datasets/person.csv')

    # Imputing blanks in SEATING_POSITION
    # The only blanks in SEATING_POSITION are for 'drivers' in ROAD_USER_TYPE- replace these with 'D'
    filtered_person.replace(r'^\s*$', 'D', regex=True, inplace=True)

    # Replacing ambiguous or unknown values with NaN
    filtered_person.replace(['Unknown', 'Not Known', 'N/A', 'NK'], np.nan, inplace=True)

    # Creating new binary feature- IN_METAL_BOX
    in_metal_box(filtered_person)

    # Creating new binary feature- UNPROTECTED
    imputing_safety_equipment(filtered_person)

    # Creating a version with no NaN values
    filtered_person_no_nan = filtered_person
    filtered_person_no_nan = filtered_person_no_nan.dropna()

    # Saving these processed datasets
    filtered_person.to_csv('datasets/filtered_person.csv', index=False)
    filtered_person_no_nan.to_csv('datasets/filtered_person_no_nan.csv', index=False)

# --- Function to create the IN_METAL_BOX feature ---
def in_metal_box(filtered_person):
    filtered_vehicle_csv = pd.read_csv('datasets/filtered_vehicle_new.csv')

    # Merging to bring VEHICLE_TYPE info into the person dataset
    merged = pd.merge(
        filtered_person,
        filtered_vehicle_csv[['ACCIDENT_NO', 'VEHICLE_ID', 'VEHICLE_TYPE']],
        on=['ACCIDENT_NO', 'VEHICLE_ID'],
        how='left'
    )

    # Defining which vehicles are considered 'exposed' or 'encased'
    encased_vehicle = {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 19, 27, 60, 61, 62, 63, 71, 72}
    exposed_vehicle = {10, 11, 12, 13, 14, 20}

    def classify_box(vehicle_type):
        if vehicle_type in encased_vehicle:
            return 1
        elif vehicle_type in exposed_vehicle:
            return 0
        else:
            return 2

    merged['IN_METAL_BOX'] = merged['VEHICLE_TYPE'].apply(classify_box)
    filtered_person['IN_METAL_BOX'] = merged['IN_METAL_BOX']

    # If unknown values have road type user of 'Passenger', they are in an enclosed vehicle
    filtered_person.loc[
        (filtered_person['IN_METAL_BOX'] == 2) &
        (filtered_person['ROAD_USER_TYPE_DESC'] == 'Passengers'),
        'IN_METAL_BOX'
    ] = 1

    # Else for the following road user types, they would be in an exposed vehicle
    filtered_person.loc[
        (filtered_person['IN_METAL_BOX'] == 2) &
        (filtered_person['ROAD_USER_TYPE_DESC'].isin([
            'Bicyclists', 'E-scooter Rider', 'Pedestrians', 'Motorcyclists', 'Pillion Passengers'
        ])),
        'IN_METAL_BOX'
    ] = 0

    # Then applying random distribution to the remaining unknown values
    randomly_imputing_in_metal_box(filtered_person)

# --- Function that applies random distribution to unknown values in 'IN_METAL_BOX' ---
def randomly_imputing_in_metal_box(filtered_person):
    is_in_metal_box = filter_out_value(filtered_person, 'IN_METAL_BOX', 1)
    not_in_metal_box = filter_out_value(filtered_person, 'IN_METAL_BOX', 0)

    # Count known values
    num_encased = is_in_metal_box.shape[0]
    num_exposed = not_in_metal_box.shape[0]
    num_total = num_encased + num_exposed

    # Compute probabilities based on known ratios
    probability_ex = num_exposed / num_total
    probability_en = num_encased / num_total

    # Find unknown values to impute
    unknown_mask = filtered_person['IN_METAL_BOX'] == 2

    # Randomly assigned based on distribution
    imputed_values = np.random.choice(
        [0, 1],
        size=unknown_mask.sum(),
        p=[probability_ex, probability_en]
    )

    # Apply imputed values only to unknown rows
    filtered_person.loc[unknown_mask, 'IN_METAL_BOX'] = imputed_values

# --- Create UNPROTECTED feature from HELMET_BELT_WORN ---
def imputing_safety_equipment(filtered_person):
    # Normalise missing entries and convert to numeric
    filtered_person['HELMET_BELT_WORN'] = filtered_person['HELMET_BELT_WORN'].replace(['', ' ', 'nan'], pd.NA)
    filtered_person['HELMET_BELT_WORN'] = pd.to_numeric(filtered_person['HELMET_BELT_WORN'], errors='coerce')

    # Classifying the 'protection status'
    filtered_person['UNPROTECTED'] = filtered_person['HELMET_BELT_WORN'].apply(
        lambda x: 0 if x in [1, 3, 6] # Wore safety equipment
        else 1 if x in [2, 4, 5, 7, 8] # Did not wear safety equipment
        else 2 # Unknown
    )

    # Randomly imputing unknown safety equipment usage IN an enclosed vehicle
    random_imputation(filtered_person, 1)

    # Randomly imputing unknown safety equipment usage NOT in an enclosed vehicle
    random_imputation(filtered_person, 0)

# --- Random imputation of UNPROTECTED values based on group exposure (encased or exposed) ---
def random_imputation(filtered_person, is_encased):
    # Cleaning empty or string-based missing entries
    filtered_person['HELMET_BELT_WORN'] = filtered_person['HELMET_BELT_WORN'].replace(['', ' ', 'nan'], pd.NA)
    filtered_person['HELMET_BELT_WORN'] = pd.to_numeric(filtered_person['HELMET_BELT_WORN'], errors='coerce')

    # Selecting records that require random imputation
    mask = (filtered_person['UNPROTECTED'] == 2) & (filtered_person['IN_METAL_BOX'] == is_encased)

    # Splitting known protected/ unprotected examples within the same group
    safety_worn = filter_out_value(filter_out_value(filtered_person, 'UNPROTECTED', 0),
                                   'IN_METAL_BOX', is_encased)
    safety_not_worn = filter_out_value(filter_out_value(filtered_person, 'UNPROTECTED', 1),
                                       'IN_METAL_BOX', is_encased)

    #  Calculating number of instances for safety equipment worn/ not worn for 'is_encased'
    num_safety_worn = safety_worn['UNPROTECTED'].shape[0]
    num_safety_not_worn = safety_not_worn['UNPROTECTED'].shape[0]
    num_total = num_safety_worn + num_safety_not_worn

    #   Estimating probabilities of person wearing/ not wearing safety equipment within 'is_encased' vehicle
    probability_0 = num_safety_worn / num_total
    probability_1 = num_safety_not_worn / num_total

    # Randomly assign worn (1), not worn (0)
    imputed_values = np.random.choice([1, 0], size=mask.sum(), p=[probability_0, probability_1])
    filtered_person.loc[mask, 'UNPROTECTED'] = imputed_values





