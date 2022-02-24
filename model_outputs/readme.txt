Set base path for this directory in terminal/cmd


Pre-Compiled for Windows :

Execute commands -> 

1. trec_eval -m map -m P.5 -m ndcg test_qrels.txt bm25_output.out 
2. trec_eval -m map -m P.5 -m ndcg test_qrels.txt vector_space_output.out
3. trec_eval -m map -m P.5 -m ndcg test_qrels.txt languagemodel_output.out