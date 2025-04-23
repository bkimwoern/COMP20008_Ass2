import pandas as pd

def preprocessing():
    person_csv = pd.read_csv('datasets/datasets-100/person-100.csv')

    fatalities_csv = person_csv[person_csv['INJ_LEVEL'] == 1]
    serious_injury_csv = person_csv[person_csv['INJ_LEVEL'] == 2]
    other_injury_csv = person_csv[person_csv['INJ_LEVEL'] == 3]
    not_injured_csv = person_csv[person_csv['INJ_LEVEL'] == 4]

    #print(fatalities_csv.head(100))
    #print(serious_injury_csv.head(100))
    #print(other_injury_csv.head(100))
    #print(not_injured_csv.head(100))

