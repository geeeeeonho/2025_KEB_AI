import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor  # 적용모델 : K 최근접 이웃 회귀 모델
from sklearn.model_selection import train_test_split  # 훈련 / 검증 셋트 분할 함수

# 나이에 따른 생존율 계산

titanic = sns.load_dataset('titanic')  # 데이터 로딩
median_age = titanic['age'].median()  # 나이 중앙값 산출
titanic_fill_row = titanic.fillna({'age' : median_age})  # 결측치 처리

X = titanic_fill_row[['age']]  # 독립 변수 설정
y = titanic_fill_row[['survived']]  # 종속 변수 설정

# 훈련/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 선택
model = KNeighborsRegressor(n_neighbors=5)

# K 최근접 이웃 회귀 모델 훈련
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)  # 검증 셋트를 인수로 예측
#예측 강화
y_pred = np.where(y_pred < 0.5, 0, 1)  # 0.5 미만은 0, 이상은 1

# 시각화
plt.figure(figsize=(5, 2))
plt.scatter(X_test, y_test, color='blue', label='Real')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('KNeighborsRegressor: Real vs Predicted')
plt.xlabel('Age')
plt.ylabel('Survivied')
plt.show()

# # 맞춘 경우 'purple', 틀린 경우 'red'로 색상 설정
# colors = np.where(y_test.values == y_pred, 'purple', 'red')
# plt.figure(figsize=(5, 2))
# plt.scatter(X_test, y_test, color='blue', label='Real')
# plt.scatter(X_test, y_pred, color=colors, label='Predictions')
# plt.xlabel('Age')
# plt.ylabel('Survivied')
# plt.show()