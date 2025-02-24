import numpy as np
import pandas as pd

df = pd.DataFrame(
        [[4,5,6],
        [7,8,9],
        [10,11,12]], index=[1,2,3], columns=['a','b','c']
)
print(df)