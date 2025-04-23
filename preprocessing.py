import pandas as pd

def preprocessing():
    accident_100_csv = pd.read_csv('datasets/datasets-100/accident-100.csv')
    print(accident_100_csv.head(100))

