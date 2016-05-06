from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle

df = pickle.load(open('descriptions.pkl', 'rb'))

# Taken from http://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph
vect = CountVectorizer(ngram_range=(1,2), stop_words='english')
pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
X = pipeline.fit_transform(df.description.values).todense()

pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
plt.scatter(data2D[:,0], data2D[:,1])
plt.show()
