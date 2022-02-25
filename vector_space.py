from contextlib import redirect_stdout
from math import sqrt


def get_similarity_matrix(document_tf_idf, query_tf_idf):
    similarity_matrix = dict()
    numerator = 0
    denominator_a = 0
    denominator_b = 0
    for document, value in document_tf_idf.items():
        for doc_key, doc_tf_idf_val in document_tf_idf[document].items():
            denominator_a += doc_tf_idf_val * doc_tf_idf_val
        for term, val in query_tf_idf.items():
            if term in document_tf_idf[document]:
                numerator += document_tf_idf[document][term] * query_tf_idf[term]
            denominator_b += query_tf_idf[term] * query_tf_idf[term]
        if denominator_a != 0 and denominator_b != 0:
            similarity_matrix.update({document: numerator / (sqrt(denominator_a) * sqrt(denominator_b))})
            numerator = 0
            denominator_a = 0
            denominator_b = 0
    return similarity_matrix


def generate_vector_space_output(document_tf_idf, query_tf_idf):
    with open('model_outputs/vector_space_output.out', 'w') as f:
        with redirect_stdout(f):
            for query_id, vector_value in query_tf_idf.items():
                similarity_matrix = get_similarity_matrix(document_tf_idf, query_tf_idf[query_id])
                with redirect_stdout(f):
                    for element in range(len(similarity_matrix)):
                        if element + 1 > 1000:
                            break
                        else:
                            document_id = max(similarity_matrix, key=lambda x: similarity_matrix[x])
                            print(str(query_id), str(1), str(document_id),  str(element + 1), str(similarity_matrix[document_id]), 'run1')
                            similarity_matrix.pop(document_id)


