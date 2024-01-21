import pandas as pd
import os
import warnings
import numpy as np
warnings.filterwarnings(action='ignore')
import random

from itertools import chain
from collections import Counter

problem = pd.read_csv('./problemList.csv',encoding="ISO-8859-1")

problem_level_seq_list = []

idx=problem[problem['level']==0].index
problem.drop(idx,inplace=True)
group_df = problem.groupby('level')

for level, df in group_df:
    problem_level_seq_list.append(df['problemId'].tolist())

print(len(problem_level_seq_list))
for i in problem_level_seq_list:
  print(i)

  problem_tag_seq_list = []

tag_list = []

def get_preprocessing_tags(tags):
    global tag_list
    # tags = eval(tags)
    if type(tags) != 'float':
      tags = str(tags).split(',')
      tags.pop()
      tags = [tag for tag in tags]
      tag_list += tags
    return tags

problem['preprocessing_tags'] = problem['key'].apply(lambda x : get_preprocessing_tags(x))
tag_list = list(set(tag_list))
problem_num_list = problem['problemId'].tolist()

tag_df = pd.DataFrame(data = np.zeros((len(problem_num_list), len(tag_list))), columns = tag_list, index = problem_num_list)

for df in problem.iloc:
    tag_df.loc[df['problemId'], df['preprocessing_tags']] = 1

for tag in tag_list:
    problem_tag_seq = tag_df[tag_df[tag] == 1].index.tolist()
    problem_tag_seq_list.append(problem_tag_seq)

len(problem_tag_seq_list)

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
import datetime

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 1
        self.loss_to_be_subed = 0
        self.loss_now = 987654321

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed

        self.loss_to_be_subed = loss
        print(loss_now, self.loss_now)
        if loss_now < self.loss_now:
            self.loss_now = loss_now
            # model.save(os.path.join(MODEL_PATH, 'clean-Word2Vec-CBOW-problem_association_seq-problem_level_seq-problem_tag_seq-vs128.model'))
            print(f'Loss after epoch {self.epoch}: {loss_now}')
            # print('Model 저장')
        self.epoch += 1

start = datetime.datetime.now()

sentences = problem_level_seq_list+problem_tag_seq_list

model = Word2Vec(
                sentences = sentences,
                seed = 22,
                epochs = 200,
                min_count = 1,
                vector_size = 128,
                sg = 0,
                negative = 10,
                window = 987654321,
                compute_loss = True,
                callbacks=[callback()],
                 )

print("Time passed: " + str(datetime.datetime.now() - start))

a=model.save('savedmodel.pkl')
