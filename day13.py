import pandas as pd

data = [1,7,5,2,8,3,6,4]
bins=[0,3,6,9]

labels = ['low','min','high']

#데이터를 bins카테고리 안에 넣기
cat = pd.cut(data,bins,labels=labels,right=False) #False: 끝 우측 값 불허
print(cat)