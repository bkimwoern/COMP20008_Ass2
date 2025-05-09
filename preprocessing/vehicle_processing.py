import pandas as pd

from .utils import filter_out_value


def process_vehicle_csv():
    filtered_vehicle_csv = pd.read_csv('datasets/filtered_vehicle.csv')

    process_vehicle_body_type(filtered_vehicle_csv)
    process_vehicle_type(filtered_vehicle_csv)

    filtered_vehicle_csv.to_csv('datasets/filtered_vehicle_new.csv', index=False)

def process_vehicle_body_type(filtered_vehicle):
    vehicle_body_map = {
        'OF SHD': 'OF/SHD',
        'SEDAN' : 'SED',
    }
    filtered_vehicle['VEHICLE_BODY_STYLE'] = filtered_vehicle['VEHICLE_BODY_STYLE'].replace(vehicle_body_map)

def process_vehicle_type(filtered_vehicle_csv):
    # --- Map for 'Not Applicable' values in VEHICLE_TYPE_DESC ---
    vehicle_body_type_map = {
        'P MVR': 'Prime Mover (No of Trailers Unknown)',
        'WAGON': 'Station Wagon',
        'S WAG': 'Station Wagon',
        'TRAY': 'Utility',
        'TIPPER': 'Heavy Vehicle',
        'CONVRT': 'Prime Mover (No of Trailers Unknown)',
        'SED': 'Car',
        'SKIP C': 'Heavy Vehicle',
    }

    vehicle_type_map = {
        'Prime Mover (No of Trailers Unknown)': 6,
        'Station Wagon': 2,
        'Utility': 4,
        'Heavy Vehicle': 72,
        'Car': 1,
    }

    body_mask = filtered_vehicle_csv['VEHICLE_TYPE_DESC'] == 'Not Applicable'
    filtered_vehicle_csv.loc[body_mask, 'VEHICLE_TYPE_DESC'] = (
        filtered_vehicle_csv.loc[body_mask, 'VEHICLE_BODY_STYLE']
        .map(vehicle_body_type_map)
        .fillna(filtered_vehicle_csv.loc[body_mask, 'VEHICLE_TYPE_DESC'])
    )

    # numbers (vehicle_Type) not being updated correctly
    type_mask = filtered_vehicle_csv['VEHICLE_TYPE'] == 18
    filtered_vehicle_csv.loc[type_mask, 'VEHICLE_TYPE'] = (
        filtered_vehicle_csv.loc[type_mask, 'VEHICLE_TYPE_DESC']
        .map(vehicle_type_map)
        .fillna(filtered_vehicle_csv.loc[type_mask, 'VEHICLE_TYPE'])
    )


