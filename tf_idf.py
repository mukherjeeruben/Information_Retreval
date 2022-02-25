from extract_load_save import get_file_list
import xml.etree.ElementTree as et
import re as regular_expression
from nltk import word_tokenize, corpus, WordNetLemmatizer
from math import log


def generate_tf_idf(directory_path, element_type):
    file_list = get_file_list(directory_path)
    lemmatizer = WordNetLemmatizer()
    term_count = dict()
    term_frequency = dict()
    inverse_document_frequency = dict()
    tf_idf = dict()
    for file in range(0, len(file_list)):
        tree = et.parse(directory_path + '\\' + file_list[file])
        root = tree.getroot()
        primary_key = root[0].text
        if element_type == 'document':
            data = " ".join([word for word in set((str(root[1].text) + str(root[2].text).lower()).split()) if
                             word not in corpus.stopwords.words('english')])
        else:
            data = " ".join([word for word in set(str(root[1].text).lower().split()) if word not in corpus.stopwords.words('english')])
        data = lemmatizer.lemmatize(regular_expression.sub('[^a-zA-Z0-9]+', ' ', data))
        toknized_words = [word for word in word_tokenize(data) if len(word) > 2]
        word_count = 0
        tf_idf[primary_key] = dict()
        for word_item in range(0, len(toknized_words)):
            stemmed_word = toknized_words[word_item]
            word_count += 1
            if stemmed_word in term_count:
                if primary_key in term_count[stemmed_word]:
                    term_count[stemmed_word][primary_key] += 1
                    term_frequency[stemmed_word][primary_key] = term_count[stemmed_word][primary_key] / word_count
                    inverse_document_frequency[stemmed_word] = log(len(file_list)/len(term_count[stemmed_word]))
                else:
                    term_count[stemmed_word][primary_key] = 1
                    term_frequency[stemmed_word][primary_key] = 1 / word_count
                    inverse_document_frequency[stemmed_word] = log(len(file_list)/len(term_count[stemmed_word]))
            else:
                term_count[stemmed_word] = dict()
                term_frequency[stemmed_word] = {}
                term_count[stemmed_word][primary_key] = 1
                term_frequency[stemmed_word][primary_key] = 1 / word_count
                inverse_document_frequency[stemmed_word] = log(len(file_list)/len(term_count[stemmed_word]))
            if stemmed_word in term_frequency:
                tf_idf[primary_key][stemmed_word] = term_frequency[stemmed_word][primary_key] * inverse_document_frequency[stemmed_word]
            else:
                tf_idf[primary_key][stemmed_word] = 0
    return tf_idf

