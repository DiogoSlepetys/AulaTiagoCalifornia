from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import pandas as pd

#open CSV file
data = pd.read_csv('precos_casa_california.csv')

def change2Float(data):
    return pd.to_numeric(data, errors='coerce')
def change2Int(data):
    return pd.to_numeric(data, errors='coerce', downcast='integer')

dataNew = data
dataNew["longitude"] = change2Float(dataNew["longitude"])
dataNew["housing_median_age"] = change2Int(dataNew["housing_median_age"])
dataNew["total_rooms"] = change2Int(dataNew["total_rooms"])
dataNew["latitude"] = change2Float(dataNew["latitude"])
dataNew["total_bedrooms"] = change2Int(dataNew["total_bedrooms"])
dataNew["population"] = change2Int(dataNew["population"])
dataNew["households"] = change2Int(dataNew["households"])
dataNew["median_income"] = change2Float(dataNew["median_income"])
dataNew["median_house_value"] = change2Float(dataNew["median_house_value"])
dataNew['ocean_proximity'] = dataNew['ocean_proximity'].replace({'<1H OCEAN': 0, 'INLAND': 1,'NEAR OCEAN':2,'NEAR BAY':3, 'ISLAND':4})
dataNew["ocean_proximity"] = change2Float(dataNew["ocean_proximity"])
dataNew = dataNew.dropna()

dataT = dataNew
#dataT['rooms_p_household'] = dataT['total_rooms']/dataT['households']
dataT['rooms_p_household'] = dataT['total_rooms'] / dataT['total_bedrooms']
dataT['bedrooms_p_rooms'] = dataT['total_bedrooms'] / dataT['total_rooms']
dataT['population_p_household'] = dataT['population'] / dataT['households']
dataT = dataT[dataT['median_house_value'] <= 1000000].reset_index(drop=True)
dataT = dataT[dataT['housing_median_age'] <= 100].reset_index(drop=True)
dataT = dataT[dataT['median_income'] <= 14].reset_index(drop=True)
dataT = dataT[dataT['population'] <= 9000]
dataT = dataT[(dataT['population_p_household'] <=10 )]
dataT = dataT[dataT['rooms_p_household'] <20 ]

# show data
# corr_matrix = dataT.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# dataT.hist(bins=30,figsize=(10,15))
# plt.show()

y = dataT['median_house_value']
x = dataT.drop(columns=['median_house_value'])
x_treino, x_teste, y_treino, y_teste = train_test_split( x, y, test_size = 0.5, random_state = 8)

model = GradientBoostingRegressor()
model.n_estimators= 500
model.max_depth = 4
model.fit(x_treino,y_treino)
score = model.score(x_teste,y_teste)
print("teste:")
print(score)
score = model.score(x_treino,y_treino)
print("treino:")
print(score)



