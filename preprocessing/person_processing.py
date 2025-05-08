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

def in_metal_box(filtered_person):
    filtered_vehicle_csv = pd.read_csv('datasets/filtered_vehicle.csv')

    enclosed_vehicle = ['Bus/Coach', ]

    #   Defaulting all values to 0 (is not in enclosed vehicle)
    #   Join person_csv with filtered_vehicle, which has description of type of vehicle, then base it off that!!!
    filtered_person['IN_METAL_BOX'] = 0


    filtered_person.loc[filtered_person['ROAD_USER_TYPE_DESC'].isin(['Drivers', 'Passengers']), 'IN_METAL_BOX'] = 1
    filtered_person.loc[filtered_person['ROAD_USER_TYPE_DESC'].isna(), 'IN_METAL_BOX'] = np.nan


def imputing_safety_equipment(filtered_person):
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

    metal = filter_out_value(filtered_person, 'IN_METAL_BOX', 1)
    print(filter_out_value(metal, 'UNPROTECTED', 2))

def random_imputation(filtered_person, is_encased):
    print('Before')
    print(filtered_person['UNPROTECTED'].value_counts())

    # --- Extracting records of unknown safety equipment usage IN an encased vehicle ---
    unknown = filter_out_value(filter_out_value(filtered_person, 'HELMET_BELT_WORN', 9.0),
                                       'IN_METAL_BOX', is_encased)

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

    #   Applying weighted random imputation to the 'UNPROTECTED' column where value is 2 (unknown if safety
    #   equipment was used)
    unknown['UNPROTECTED'] = unknown['UNPROTECTED'].apply(
        lambda x: np.random.choice([0, 1], p=[probability_0, probability_1]) if x == 2 else x
    )
    #   Assigning these imputed values back to filtered_person
    filtered_person.loc[unknown.index, 'UNPROTECTED'] = unknown['UNPROTECTED']

    print('After imputing IN_METAL_BOX =', is_encased)
    print(filtered_person['UNPROTECTED'].value_counts())
