from .vehicle_processing import process_vehicle_csv, process_vehicle_body_type
from .person_processing import process_person_csv
from .accident_processing import process_accident_csv

# --- Main preprocessing function that calls sub-functions to preprocess specific csv files ---
def preprocessing():
    process_vehicle_csv()
    process_person_csv()
    process_accident_csv()


