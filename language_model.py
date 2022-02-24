from contextlib import redirect_stdout


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









