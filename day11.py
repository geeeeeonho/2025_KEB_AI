# from sklearn.linear_model import LinearRegression
# import ghlearn as gh
from ghlearn import LinearRegression #자체 제작 예측
import pandas as pd
import matplotlib.pyplot as plt

ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values

ls.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23500, 62500, 4, 9])
plt.show()

model = LinearRegression()
model.fit(X, y)

X_new = [[31721.3]]  # ROK 2020
print(model.predict(X_new))
# LinearRegression 5.90
# KNeighborsRegressor 5.70