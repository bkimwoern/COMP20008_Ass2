import numpy as np
import pandas as pd
from .utils import filter_out_value

def process_person_csv():
    filtered_person = pd.read_csv('datasets/person.csv')

    # --- Imputing blanks in SEATING_POSITION ---
    #    The only blanks in SEATING_POSITION are for 'drivers' in ROAD_USER_TYPE
    #    Replacing blanks and 'nan's' with 'D'
    filtered_person.replace(r'^\s*$', 'D', regex=True, inplace=True)

    # --- Converting unknown variations in dataset to 'nan' ---
    filtered_person.replace(['Unknown', 'Not Known', 'N/A', 'NK', '5-Dec'], np.nan, inplace=True)

    # --- Creating new column indicating whether a person was in an enclosed vehicle ---
    in_metal_box(filtered_person)

    # --- Imputing not known values in 'HELMET_BELT_WORN' column ---
    imputing_safety_equipment(filtered_person)

    filtered_person.to_csv('datasets/filtered_person.csv', index=False)

    filtered_person_no_nan = filtered_person
    filtered_person_no_nan.dropna()
    filtered_person_no_nan.tocsv('datasets/filtered_person_no_nan.csv', index=False)

def in_metal_box(filtered_person):
    filtered_vehicle_csv = pd.read_csv('datasets/filtered_vehicle_new.csv')

    encased_vehicle = {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 19, 27, 60, 61, 62, 63, 71, 72}
    exposed_vehicle = {10, 11, 12, 13, 14, 20}

    def classify_box(vehicle_type):
        if vehicle_type in encased_vehicle:
            return 1
        elif vehicle_type in exposed_vehicle:
            return 0
        else:
            return 2

    merged = pd.merge(
        filtered_person,
        filtered_vehicle_csv[['ACCIDENT_NO', 'VEHICLE_ID', 'VEHICLE_TYPE']],
        on=['ACCIDENT_NO', 'VEHICLE_ID'],
        how='left'
    )

    merged['IN_METAL_BOX'] = merged['VEHICLE_TYPE'].apply(classify_box)
    filtered_person['IN_METAL_BOX'] = merged['IN_METAL_BOX']

    filtered_person.loc[
        (filtered_person['IN_METAL_BOX'] == 2) &
        (filtered_person['ROAD_USER_TYPE_DESC'] == 'Passengers'),
        'IN_METAL_BOX'
    ] = 1

    filtered_person.loc[
        (filtered_person['IN_METAL_BOX'] == 2) &
        (filtered_person['ROAD_USER_TYPE_DESC'].isin([
            'Bicyclists', 'E-scooter Rider', 'Pedestrians', 'Motorcyclists', 'Pillion Passengers'
        ])),
        'IN_METAL_BOX'
    ] = 0

    # NOW FOR THE REMAINING 2'S IN IN_METAL_BOX, DO PROPORTIONAL IMPUTATION
    randomly_imputing_in_metal_box(filtered_person)

def randomly_imputing_in_metal_box(filtered_person):
    is_in_metal_box = filter_out_value(filtered_person, 'IN_METAL_BOX', 1)
    not_in_metal_box = filter_out_value(filtered_person, 'IN_METAL_BOX', 0)

    num_encased = is_in_metal_box.shape[0]
    num_exposed = not_in_metal_box.shape[0]
    num_total = num_encased + num_exposed

    probability_ex = num_exposed / num_total
    probability_en = num_encased / num_total

    unknown_mask = filtered_person['IN_METAL_BOX'] == 2

    imputed_values = np.random.choice(
        [0, 1],
        size=unknown_mask.sum(),
        p=[probability_ex, probability_en]
    )

    # Apply imputed values only to unknown rows
    filtered_person.loc[unknown_mask, 'IN_METAL_BOX'] = imputed_values

def imputing_safety_equipment(filtered_person):
    filtered_person['HELMET_BELT_WORN'] = filtered_person['HELMET_BELT_WORN'].replace(['', ' ', 'nan'], pd.NA)
    filtered_person['HELMET_BELT_WORN'] = pd.to_numeric(filtered_person['HELMET_BELT_WORN'], errors='coerce')

    # --- Creating a new column 'UNPROTECTED'- 1 if no safety equipment was worn, 0 if worn ---
    filtered_person['UNPROTECTED'] = filtered_person['HELMET_BELT_WORN'].apply(
        lambda x: 0 if x in [1, 3, 6] # Wore safety equipment
        else 1 if x in [2, 4, 5, 7, 8] # Did not wear safety equipment
        else 2 # Unknown
    )
    # -- Randomly imputing unknown safety equipment usage IN an enclosed vehicle
    random_imputation(filtered_person, 1)

    # --- Randomly imputing unknown safety equipment usage NOT in an enclosed vehicle
    random_imputation(filtered_person, 0)

def random_imputation(filtered_person, is_encased):
    # Cleaning empty or string-based missing entries
    filtered_person['HELMET_BELT_WORN'] = filtered_person['HELMET_BELT_WORN'].replace(['', ' ', 'nan'], pd.NA)
    filtered_person['HELMET_BELT_WORN'] = pd.to_numeric(filtered_person['HELMET_BELT_WORN'], errors='coerce')

    # Mask for rows needing random imputation
    mask = (filtered_person['UNPROTECTED'] == 2) & (filtered_person['IN_METAL_BOX'] == 1)
    safety_worn = filter_out_value(filter_out_value(filtered_person, 'UNPROTECTED', 0),
                                   'IN_METAL_BOX', is_encased)
    safety_not_worn = filter_out_value(filter_out_value(filtered_person, 'UNPROTECTED', 1),
                                       'IN_METAL_BOX', is_encased)

    # --- Using weighted random imputation to impute 1 or 0 in 'UNPROTECTED', where use of safety equipment is unknown ---
    num_safety_worn = safety_worn['HELMET_BELT_WORN'].shape[0]
    num_safety_not_worn = safety_not_worn['HELMET_BELT_WORN'].shape[0]
    num_total = num_safety_worn + num_safety_not_worn

    #   Finding probabilities of person wearing/ not wearing safety equipment within an encased vehicle
    probability_0 = num_safety_worn / num_total
    probability_1 = num_safety_not_worn / num_total

    # Randomly assign worn (1), not worn (0)
    imputed_values = np.random.choice([1, 0], size=mask.sum(), p=[probability_0, probability_1])
    filtered_person.loc[mask, 'HELMET_BELT_WORN'] = imputed_values

    # Recalculate UNPROTECTED after imputation
    filtered_person['UNPROTECTED'] = filtered_person['HELMET_BELT_WORN'].apply(
        lambda x: 0 if x in [1, 3, 6] else 1 if pd.notna(x) else 2
    )




