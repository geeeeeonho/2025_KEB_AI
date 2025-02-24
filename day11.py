import matplotlib.pyplot as plt                   #그래픽 라이브러리
import pandas as pd                               #판다스
from sklearn.linear_model import LinearRegression #모델 불러오기
from sklearn.neighbors import KNeighborsRegressor

#read_csv : pandas에서 기본 제공하는 읽기
#데이터 해당 링크의 데이터 읽어오기
ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
#print(ls)
X=ls[["GDP per capita (USD)"]].values   #x축 설정
Y=ls[["Life satisfaction"]].values        #y축 설정

#데이터를 그래프로 나타냄
ls.plot(kind='scatter',grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23500, 62500, 4, 9]) #axis([x최소범위,x최대범위,y최소범위,y최대범위])
plt.show() #그래프를 보여줌

#model = LinearRegression() #선형 회귀 모델 제작
model = KNeighborsRegressor(n_neighbors=3)  #최근접 이웃 회귀 모델 제작
model.fit(X,Y)

X_new = [[37655.2]]
print('37655.2->',model.predict(X_new)) #predict(x) x를 이용한 예측모델