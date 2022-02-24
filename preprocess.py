from extract_load_save import get_file_list
import xml.etree.ElementTree as et
import re as regular_expression
from nltk import word_tokenize, corpus, stem


def preprocess(directory_path, element_type):
    word_map = dict()
    stemmer = stem.PorterStemmer()
    file_list = get_file_list(directory_path)
    for file in range(0, len(file_list)):
        tree = et.parse(directory_path + '\\' + file_list[file])
        root = tree.getroot()
        primary_key = root[0].text
        if element_type == 'document':
            data = " ".join([word for word in set((str(root[1].text) + str(root[2].text).lower()).split()) if
                                 word not in corpus.stopwords.words('english')])
        else:
            data = " ".join([word for word in set(str(root[1].text).lower().split()) if
                             word not in corpus.stopwords.words('english')])
        data = stemmer.stem(regular_expression.sub('[^a-zA-Z0-9]+', ' ', data))
        word_map[primary_key] = [word for word in word_tokenize(data) if len(word) > 2]
    average = sum(len(sentence) for sentence in [word_map[key] for key in word_map])/len(word_map)
    return word_map, average