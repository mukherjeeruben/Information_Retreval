# from extract_load_save import get_file_list
# import xml.etree.ElementTree as et
# import re as regular_expression
# from nltk import word_tokenize, corpus, stem
from math import log
from contextlib import redirect_stdout


# def preprocess_bm25(directory_path, element_type):
#     word_map = dict()
#     stemmer = stem.PorterStemmer()
#     file_list = get_file_list(directory_path)
#     for file in range(0, len(file_list)):
#         tree = et.parse(directory_path + '\\' + file_list[file])
#         root = tree.getroot()
#         primary_key = root[0].text
#         if element_type == 'document':
#             data = " ".join([word for word in set((str(root[1].text) + str(root[2].text).lower()).split()) if
#                                  word not in corpus.stopwords.words('english')])
#         else:
#             data = " ".join([word for word in set(str(root[1].text).lower().split()) if
#                              word not in corpus.stopwords.words('english')])
#         data = stemmer.stem(regular_expression.sub('[^a-zA-Z0-9]+', ' ', data))
#         word_map[primary_key] = [word for word in word_tokenize(data) if len(word) > 2]
#     average = sum(len(sentence) for sentence in [word_map[key] for key in word_map])/len(word_map)
#     return word_map, average


def bm25(word, doc_list, sentence, average, k=1.2, b=0.75):
    frequency = sentence.count(word)
    if frequency > 0:
        term_frequency = (frequency * (k + 1)) / (frequency + k * (1 - b + b * len(sentence)) / average)
        n_q = sum([1 for doc in doc_list if word in doc])
        idf = log((len(doc_list) - n_q + 0.5) / (n_q + 0.5)) + 1
        return round(term_frequency * idf, 4)
    else:
        return frequency


def generate_bm25_vectors(doc_word_map, query_word_map, average):
    vectors = list()
    for query_id, sentence in query_word_map.items():
        word_map = dict()
        word_map[query_id] = dict()
        for document_id, text in doc_word_map.items():
            bm25_score = 0
            for vocab in sentence:
                bm25_score += bm25(vocab, doc_word_map, text, average)
                word_map[query_id].update({document_id: bm25_score})
        vectors.append(word_map)
    return vectors


def generate_bm25_output(bm25vectors):
    with open('bm25_output.out', 'w') as f:
        with redirect_stdout(f):
            for terms in bm25vectors:
                for query_id, v in terms.items():
                    rank = 0
                    doc_set = {k: v for k, v in reversed(sorted(v.items(), key=lambda item: item[1]))}
                    for doc_key, doc_vals in doc_set.items():
                        rank += 1
                        if rank > 1000:
                            break
                        else:
                            print(query_id, str(1), doc_key, rank, doc_vals, 'run1')









