import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('~/Downloads/mdst-getstarted/ums/data/ums_viz.csv')
df_perf = pd.read_pickle('descriptions.pkl')
df_perf.columns.values[0] = 'perf_name'

df['perf_name'] = df['perf_name'].apply(lambda x: x.strip())
df_perf['perf_name'] = df_perf['perf_name'].apply(lambda x: x.strip())

df['per_seat'] = df['tck_amt']/df['num_seats']
df_group = df.groupby('perf_name')

# mean ticket amount, number of seats, price per seat
df_tmp = df_group['tck_amt', 'num_seats', 'per_seat'].mean().reset_index()
df_tmp = df_tmp.rename(columns = {'tck_amt': 'mean_tck_amt', 'num_seats': 'mean_num_seats', 'per_seat': 'mean_per_seat'})

df_perf = pd.merge(df_perf, df_tmp, on='perf_name')

# max ticket amount, number of seats, price per seat
df_tmp = df_group['tck_amt', 'num_seats', 'per_seat'].max().reset_index()
df_tmp = df_tmp.rename(columns = {'tck_amt': 'max_tck_amt', 'num_seats': 'max_num_seats', 'per_seat': 'max_per_seat'})
df_perf = pd.merge(df_perf, df_tmp, on='perf_name')

# total number of seats
df_tmp = df_group['num_seats'].count().reset_index()
df_tmp = df_tmp.rename(columns = {'num_seats': 'count_tck_amt'})
df_perf = pd.merge(df_perf, df_tmp, on='perf_name')

# proportion of student tickets
df['sprop'] = df['price_type_group'] == 'Student Prices'
df_group = df.groupby('perf_name')

df_tmp = df_group['sprop'].mean().reset_index()
df_perf = pd.merge(df_perf, df_tmp, on='perf_name')

df2 = pd.read_pickle('descriptions.pkl')
print(df2.head())

vect = CountVectorizer(ngram_range=(1,2), stop_words = ENGLISH_STOP_WORDS)
X = vect.fit_transform(df2.description.values)

lda = LatentDirichletAllocation(n_topics=5,max_iter = 100)
y = lda.fit_transform(X)

doc_topics = []
num_documents = y.shape[0]
for doc in range(num_documents):
    t = y[doc, :].argmax()
    doc_topics.append([df2.title[doc], t, df_perf.sprop[doc]])

df_sprop = pd.DataFrame(np.array(doc_topics), columns=['perf_name','topic_num','sprop'])
df_sprop.topic_num = df_sprop.topic_num.astype(dtype = np.int16)
print(df_sprop.head())

df_sprop['sprop'] = pd.to_numeric(df_sprop['sprop'], errors='coerce')
topic_range = range(5)
sns.factorplot(x='topic_num', y='sprop', data=df_sprop, kind='bar', palette='muted', size=6, aspect=1.5, order=topic_range)
plt.ylim(0, .2)
sns.plt.show()