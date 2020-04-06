import pandas as pd
from numpy import isnan

data = pd.read_csv('cases.csv', encoding = "ISO-8859-1")

country_codes = list(data.countryterritoryCode.unique())

for code in country_codes:
    if isinstance(code, str):
        print (code,data[data.countryterritoryCode == code].iloc[0].countriesAndTerritories)
