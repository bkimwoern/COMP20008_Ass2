import pandas as pd

def process_accident_csv():
    filtered_accident = pd.read_csv('datasets/accident.csv')

    # --- Normalising date values in filtered_accident_csv ---
    """
    filtered_accident['ACCIDENT_DATE'] = pd.to_datetime(filtered_accident['ACCIDENT_DATE'],
                                                        dayfirst=True,
                                                     errors='coerce')
    """
    # --- Preprocessing accident_csv ---
    #   Adding a public holiday boolean column to filtered_accident csv
    public_holiday_column(filtered_accident)
    #   Adding a night or day column to filtered_accident csv
    night_day_column(filtered_accident)
    #   Fixing incorrect DAY_OF_WEEK values in filtered_accident csv
    day_of_week(filtered_accident)
    # Adding a column that indicates whether accident occurred at an intersection (1) or not (0)
    at_intersection(filtered_accident)

    filtered_accident_no_nan = filtered_accident
    filtered_accident_no_nan.dropna()

    filtered_accident.to_csv('datasets/filtered_accident.csv', index=False)

    # Creating version without nan values
    filtered_accident_no_nan.to_csv('datasets/filtered_accident_no_nan.csv', index=False)

def public_holiday_column(filtered_accident):
    # --- Normalising values in public_holiday_csv
    public_holiday_csv = pd.read_csv('datasets/public_holiday_2012-2024.csv')
    public_holiday_csv['Date'] = pd.to_datetime(public_holiday_csv['Date'], format='%d/%m/%Y')
    public_holiday_csv['National_holiday'] = public_holiday_csv['National_holiday'].astype(bool)

    # --- Extracting dates from accident_csv that fall on national holidays or public holidays in Melbourne ---
    holiday_dates = public_holiday_csv.loc[
        (public_holiday_csv['National_holiday'] == True) | (public_holiday_csv['Melbourne'] == 1),
        'Date'
    ]

    # --- Creating a new column in filtered_accident called PUBLIC_HOLIDAY
    #   Initialising everything to false
    filtered_accident['PUBLIC_HOLIDAY'] = 0
    #   Dates that fall on national holidays are set to 1
    filtered_accident.loc[filtered_accident['ACCIDENT_DATE'].isin(holiday_dates), 'PUBLIC_HOLIDAY'] = 1

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

    #   Correcting only the mismatched rows within the dataset
    filtered_accident.loc[filtered_accident['DAY_OF_WEEK'] != expected, 'DAY_OF_WEEK'] = expected

def at_intersection(filtered_accident):
    filtered_accident['AT_INTERSECTION'] = (filtered_accident['ROAD_GEOMETRY']
                                            .apply(lambda x: 1 if x in [1,2,3,4] else 0))