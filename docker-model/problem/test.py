import pandas as pd
import json
from gensim.models import Word2Vec


load_model= Word2Vec.load("savedmodel.pkl")
problem = pd.read_csv('problemList.csv',encoding="ISO-8859-1")


def handler(event, context):
 
 data = json.loads(event['body'])
 values = data["solvedRecentId"]

 listed=[]
 listed.append(values)

 li=load_model.predict_output_word(listed,topn=15)

 problemlists=[]
 for i in range(len(li)):
  newscore=problem['acceptedUserCount'][i]*li[i][1]
  problemlists.append((li[i][0],newscore))


 sorted_list = sorted(problemlists, key=lambda x: x[1], reverse=True)
 results = [item[0] for item in sorted_list]


 return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'result': results})
        }
    

