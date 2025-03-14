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
    def __init__(self, n_neighbors=5):  # default neighbor
        self.n_neighbors = n_neighbors


    def fit(self, X_train, y_train):
        """
        learning function
        :param X: independent variable (2d array format)
        :param y: dependent variable (2d array format)
        :return: void
        """
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test):
        """
        predict value for input
        :param X: new indepent variable
        :return: predict value for input (2d array format)
        """
        predictions = []
        for x_test in X_test:  # loop just one time in this example
            # d(P, Q) = sqrt((x2 - x1)^2 + (y2 - y1)^2)
            distances = np.sqrt(np.sum((x_test - self.X_train)**2, axis=1)) #모든 x들의 거리 계산
            #거리의 순서에 대한 값들을 얻을 수 있다.
            print('주어진 기준으로부터 각각의 거리\n',distances)
            indices = np.argsort(distances)[:self.n_neighbors]
            # np.argsort(distances) 제일 작은 것을 배열(distances기준)
            # 인덱스 배열에서 self.n_neighbors개만 선택
            print('선택한 x들은 :',indices)
            prediction = np.mean(self.y_train[indices])
            # 각 선택된 x 값에 대응되는 y값의 평균
            # prediction = (self.y_train[indices[0]]+self.y_train[indices[1]]+self.y_train[indices[2]]) / self.n_neighbors
            predictions.append(prediction)

            return np.array(prediction).reshape(-1, 1)
