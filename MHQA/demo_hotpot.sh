python demo.py \
--graph_retriever_path models/hotpot_models/graph_retriever_path/pytorch_model.bin \
--reader_path models/hotpot_models/reader \
--sequential_sentence_selector_path models/hotpot_models/sequential_sentence_selector/pytorch_model.bin \
--tfidf_path models/hotpot_models/tfidf_retriever/wiki_open_full_new_db_intro_only-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz \
--db_path models/hotpot_models/wiki_db/wiki_abst_only_hotpotqa_w_original_title.db \
--bert_model_sequential_sentence_selector bert-large-uncased --do_lower_case \
--tfidf_limit 500 --eval_batch_size 4 --pruning_by_links --beam_graph_retriever 8 \
--beam_sequential_sentence_selector 8 --max_para_num 2000