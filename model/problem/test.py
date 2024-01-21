import random
import pandas as pd
from gensim.models import Word2Vec

problem = pd.read_csv('./problemList.csv',encoding="ISO-8859-1")


i=random.randint(1000,31064)
listed=[]
listed.append(i)
if i in problem['problemId'].values:
  savedmodel=Word2Vec.load('./savedmodel.pkl')
  li=savedmodel.predict_output_word(listed,topn=10)
  print(listed)
  print(i,li)
else:
  print("error")

problemlists=[]
for i in range(len(li)):
 newscore=problem['acceptedUserCount'][i]*li[i][1]
 problemlists.append((li[i][0],newscore))

sorted_list = sorted(problemlists, key=lambda x: x[1], reverse=True)

print(sorted_list)