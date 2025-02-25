import numpy as np  #넘파이
import pandas as pd #판다스
from sklearn.impute import SimpleImputer

#임의로 결측치를 만들기
#결측치 값을 산술평균으로 채워넣는 다양한 방법을 적용
df = pd.DataFrame(
    {
        'A':[1 , 2 , np.nan , 4],
        'B':[np.nan , 12 , 3 , 4],
        'C':[1 , 2 , 3 , 4],
    }
)
"""
     A     B  C
0  1.0   NaN  1
1  2.0  12.0  2
2  NaN   3.0  3
3  4.0   4.0  4
"""

#1.imputer 사용 - 평균으로 채우기
i = SimpleImputer(strategy='mean')  #평균으로 채우기
df[['A','B']] = i.fit_transform(df[['A','B']])
print(df)