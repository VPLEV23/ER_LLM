from re import sub
from gensim.utils import simple_preprocess
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def test_similarity(folder, out_file):
    list_docs = []
    stop_words = []

    with open(out_file, "w", encoding="utf8", errors='ignore') as res:
        for file in os.listdir(folder):
            f = open(folder+file, 'r', encoding="utf8", errors="encoding")
            list_docs.append(f.read().rstrip())

        similarity_matrix=[]
        for i in range(0, len(list_docs)):
            doc = list_docs.pop(i)

            list_similarities = list(1-glove_sematic_sim(doc,list_docs, stop_words))
            list_similarities.insert(i,1.0)
            res.write(','.join(str(s) for s in list_similarities)+'\n')
            similarity_matrix.append(list_similarities)
            list_docs.insert(i,doc)




def preprocess(doc,stopwords):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]



def glove_sematic_sim(query_string,documents,stopwords):

    # Preprocess the documents, including the query.txt string
    corpus = [preprocess(document,stopwords) for document in documents]
    query = preprocess(query_string,stopwords)

    # Load the model: this is a big file, can take a while to download and open
    glove = api.load("glove-wiki-gigaword-50")
    similarity_index = WordEmbeddingSimilarityIndex(glove)

    # Build the term dictionary, TF-idf model
    dictionary = Dictionary(corpus+[query])
    tfidf = TfidfModel(dictionary=dictionary)

    # Create the term similarity matrix.
    similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

    query_tf = tfidf[dictionary.doc2bow(query)]
    index = SoftCosineSimilarity(
                tfidf[[dictionary.doc2bow(document) for document in corpus]],
                similarity_matrix)

    doc_similarity_scores = index[query_tf]

    # Output the sorted similarity scores and documents
    sorted_indexes = np.argsort(doc_similarity_scores)[::-1]

    # for idx in sorted_indexes:
    #     print(f'{idx} \t {doc_similarity_scores[idx]:0.3f} \t {documents[idx]}')

    return doc_similarity_scores


def calculate_clusters(train_matrix):
    train = np.array(train_matrix)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(train)
    print(model.n_clusters_)
    return model




