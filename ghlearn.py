import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None  # weight
        self.intercept = None  # bias


    def fit(self, X, y):    #함수 제작
        """
        learning function
        :param X: independent variable (2d array format)
        :param y: dependent variable (2d array format)
        :return: void
        """
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        denominator = np.sum(pow(X-X_mean, 2))
        numerator = np.sum((X-X_mean)*(y-y_mean))

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * X_mean)


    def predict(self, X) -> np.ndarray: #예측 하기
        """
        predict value for input
        :param X: new indepent variable
        :return: predict value for input (2d array format)
        """
        return self.slope * np.array(X) + self.intercept


class KNeighborsRegressor:
    def __init__(self, n_neighbors=None):
        self.n_neighbors = n_neighbors
        self.X = None
        self.Y = None


    def fit(self, X, Y):    #함수 제작
        self.X = X.flatten()    #그래프 2차원화
        self.Y = Y.flatten()    #그래프 2차원화


    def predict(self, X_new): #예측 하기
        predictions = []
        for x in X_new:
            distances = np.abs(self.X - x) #각 거리 계산
            nearest_indices = np.argsort(distances)[:self.n_neighbors] #가장 가까운 n개 선택 #위의 생성에서 지정됨
            nearest_values = self.Y[nearest_indices] #각각의 X에 해당하는 Y 값 가져오기
            predictions.append(np.mean(nearest_values)) #평균 계산
        return np.array(predictions)