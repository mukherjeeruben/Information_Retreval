import config
from tf_idf import generate_tf_idf
from vector_space import generate_vector_space_output
from time_period import time_stamp
from bm25 import generate_bm25_vectors, generate_bm25_output
from language_model import generate_language_model_vectors, generate_languagemodel_output
from preprocess import preprocess


def vector_space_model():
    print('Vector Space Model')
    print('Calculate TF-IDF for Documents')
    time_stamp(stage='start')
    document_tf_idf = generate_tf_idf(directory_path=config.document_directory, element_type='document')
    time_stamp(stage='end')

    print('Calculate TF-IDF for Queries')
    time_stamp(stage='start')
    query_tf_idf = generate_tf_idf(directory_path=config.query_directory, element_type='query')
    time_stamp(stage='end')

    print('Calculate and Generate Vector Space Output File')
    time_stamp(stage='start')
    generate_vector_space_output(document_tf_idf, query_tf_idf)
    time_stamp(stage='end')


def bm25():
    print('Bm25')
    time_stamp(stage='start')
    doc_word_map, doc_average = preprocess(directory_path=config.document_directory, element_type='document')
    query_word_map, query_average = preprocess(directory_path=config.query_directory, element_type='query')
    bm25vectors = generate_bm25_vectors(doc_word_map, query_word_map, doc_average)
    generate_bm25_output(bm25vectors)
    time_stamp(stage='end')


def unigram_language_model():
    print('Language Model')
    time_stamp(stage='start')
    doc_word_map, doc_average = preprocess(directory_path=config.document_directory, element_type='document')
    query_word_map, query_average = preprocess(directory_path=config.query_directory, element_type='query')
    language_model_vectors = generate_language_model_vectors(doc_word_map, query_word_map)
    generate_languagemodel_output(language_model_vectors)
    time_stamp(stage='end')


if __name__ == '__main__':
    try:
        print("Enter\n 1 for Vector Space Model\n 2 for BM25 \n 3 for Unigram Language Model")
        val = int(input("Enter the model: "))
        if val == 1:
            vector_space_model()
        elif val == 2:
            bm25()
        else:
            unigram_language_model()
    except Exception as exp_msg:
        print(exp_msg)



