import pandas as pd

def process_vehicle_csv():
    filtered_vehicle_csv = pd.read_csv('datasets/filtered_vehicle.csv')
    process_vehicle_body_type(filtered_vehicle_csv)
    filtered_vehicle_csv.to_csv('datasets/filtered_vehicle_new.csv', index=False)

def process_vehicle_body_type(filtered_vehicle):
    vehicle_body_map = {
        'OF SHD': 'OF/SHD',
        'SEDAN' : 'SED',
    }
    filtered_vehicle['VEHICLE_BODY_STYLE'] = filtered_vehicle['VEHICLE_BODY_STYLE'].replace(vehicle_body_map)

#def process_vehicle_type(filtered_vehicle_csv):
    # Not applicable vehicle_body_styles:
        # P MVR
        # WAGON
        # S WAG
        # TRAY
        # WAGON
        # TRAY
        # WAGON
        # TIPPER
        # CONVRT
        # WAGON
        # SED
        # TIPPER
        # SKIP C
        # SKIP C
