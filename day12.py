import numpy as np  #넘파이
import pandas as pd #판다스
from sklearn.impute import SimpleImputer

#임의로 결측치를 만들기
df = pd.DataFrame(
    {
        'A':[1 , 2 , np.nan , 4],
        'B':[np.nan , 2 , 3 , 4],
        'C':[1 , 2 , 3 , 4],
    }
)
print(df)
