import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
 
def read_text(file_path):
    file = open(file_path, "r")
    data = file.readlines()

    sentences = [x.replace("[^a-zA-Z]", " ").split(" ") for x in data[0].split(". ")]
    sentences.pop() 
    
    return sentences

def similarity(s1, s2, stopwords=None):
    if stopwords is None:
        stopwords = []
    
    s1 = [w.lower() for w in s1]
    s2 = [w.lower() for w in s2]
 
    all_words = list(set(s1 + s2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    for w in s1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    for w in s2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_path, top_n=7):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences =  read_text(file_path)

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    print("Summarize Text: \n", ". ".join(summarize_text))


if __name__=="__main__":
    generate_summary("./test_article.txt", 2)
