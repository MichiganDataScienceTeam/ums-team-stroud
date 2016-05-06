from __future__ import unicode_literals
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import re
import datetime
import sys
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf8')
nltk.download('punkt')
stemmer = SnowballStemmer('english')

# tokenize words in docs
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

df = pd.read_pickle('descriptions.pkl')

stemmed = []
tokenized = []
for i in df.description.values:
    w_stemmed = tokenize_and_stem(i)
    stemmed.extend(w_stemmed)

    w_tokenized = tokenize_only(i)
    tokenized.extend(w_tokenized)

vf = pd.DataFrame({'words': tokenized}, index = stemmed)
print('there are ' + str(vf.shape[0]) + ' items in vf')
print(vf.head())

# compute tfidf matrix
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_df=.5, max_features=13000, min_df=.1, stop_words='english',
                             tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_mat = tfidf_vect.fit_transform(df.description.values)
print(tfidf_mat.shape)
terms = tfidf_vect.get_feature_names()
print(terms[:10])

# k-means
from sklearn.cluster import KMeans

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_mat)
clusters = km.labels_.tolist()

df_full = pd.read_csv('~/Downloads/mdst-getstarted/ums/data/ums_viz.csv', parse_dates=[1,4,12])
df.columns.values[0] = 'perf_name'
df['topic'] = clusters

df_full['perf_name'] = df_full['perf_name'].apply(lambda x: x.strip())
df['perf_name'] = df['perf_name'].apply(lambda x: x.strip())

# mean per_seat, tck_amt, num_seats
df_full['per_seat'] = df_full['tck_amt']/df_full['num_seats']
df_grp = df_full.groupby('perf_name')

df_tmp = df_grp['tck_amt', 'num_seats', 'per_seat'].mean().reset_index()
df_tmp = df_tmp.rename(columns = {'tck_amt': 'mean_tck_amt', 'num_seats': 'mean_num_seats', 'per_seat': 'mean_per_seat'})

# mean acct_age, delta_sale
df_full['acct_age'] = df_full['order_dt'] - df_full['acct_created']
df_full['delta_sale'] = df_full['perf_dt'] - df_full['order_dt']
df_full['acct_age'] = df_full['acct_age'].apply(lambda x: x.total_seconds())
df_full['delta_sale'] = df_full['delta_sale'].apply(lambda x: x.total_seconds())
df = pd.merge(df, df_tmp, on='perf_name')

df_tmp = df_grp['acct_age', 'delta_sale'].mean().reset_index()
df = pd.merge(df, df_tmp, on='perf_name')

df_grp = df.groupby('topic')
df_agg = df_grp.mean()

df_agg['acct_age'] = df_agg['acct_age'].apply(lambda x: datetime.timedelta(seconds=x))
df_agg['delta_sale'] = df_agg['delta_sale'].apply(lambda x: datetime.timedelta(seconds=x))

print df_agg.head()

# print df_agg.head()

order_centroids = km.cluster_centers_.argsort()[:,::-1]
words_by_topic = defaultdict(list)

for i in range(num_clusters):
    print "\nCluster %d words:" % i,
    for ind in order_centroids[i,:10]:
        word = vf.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')
        words_by_topic[i].append(word)
        print '%s,' % word,
    print
    print "\nCluster %d performances:" % i,
    for perf in df_grp.get_group(i)['perf_name']:
        print '%s,' % perf,
    print

for i in range(num_clusters):
    words_by_topic[i] = list(set(words_by_topic[i]))

# MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS

dist = 1 - cosine_similarity(tfidf_mat)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
pos = mds.fit_transform(dist)
xs, ys = pos[:,0], pos[:,1]

# visualize clusters
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
cluster_names = {0: ', '.join(words_by_topic[0][:5]),
                 1: ', '.join(words_by_topic[1][:5]),
                 2: ', '.join(words_by_topic[2][:5]),
                 3: ', '.join(words_by_topic[3][:5]),
                 4: ', '.join(words_by_topic[4][:5])}
df2 = pd.DataFrame(dict(x=xs, y=ys, label=clusters, perf=df['perf_name']))
groups = df2.groupby('label')

fig, ax = plt.subplots(figsize=(17,9))
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis= 'x', which='both', bottom='off', top='off', labelbottom='off')
    ax.tick_params(axis= 'y', which='both', left='off', top='off', labelleft='off')

ax.legend(loc='upper left', numpoints=1, fontsize='small')

for i in range(len(df2)):
    ax.text(df2.ix[i]['x'], df2.ix[i]['y'], df2.ix[i]['perf'], size=8)

plt.savefig('cluster_viz.png', dpi=200)
plt.show()