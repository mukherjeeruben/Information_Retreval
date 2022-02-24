from extract_load_save import get_file_list
import xml.etree.ElementTree as et
import re as regular_expression
from nltk import word_tokenize, corpus, stem
from contextlib import redirect_stdout


# def preprocess(directory_path, element_type):
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


def language_model(word, doc_text):
    frequency = doc_text.count(word)
    if frequency > 0:
        return frequency/len(doc_text)
    else:
        return frequency


def generate_language_model_vectors(doc_word_map, query_word_map):
    vectors = list()
    for query_id, sentence in query_word_map.items():
        word_map = dict()
        word_map[query_id] = dict()
        for document_id, text in doc_word_map.items():
            language_score = 0
            for vocab in sentence:
                language_score *= language_model(vocab, text)
                word_map[query_id].update({document_id: language_score})
        vectors.append(word_map)
    return vectors


def generate_languagemodel_output(bm25vectors):
    with open('languagemodel_output.out', 'w') as f:
        with redirect_stdout(f):
            for terms in bm25vectors:
                for k, v in terms.items():
                    rank = 0
                    x = {k: v for k, v in reversed(sorted(v.items(), key=lambda item: item[1]))}
                    for doc_key, doc_vals in x.items():
                        rank += 1
                        if rank > 1000:
                            break
                        else:
                            print(k, str(1), doc_key, rank, doc_vals, 'run1')









