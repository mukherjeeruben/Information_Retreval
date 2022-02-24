from math import log
from contextlib import redirect_stdout


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
    with open('model_outputs/bm25_output.out', 'w') as f:
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









