PYTHONPATH=.. python proof_search.py --deductor_path ../outputs/deductor-flant5-xl/ --selector_path ../outputs/selector-flant5-xl/ --test_data_path ../data/test/ent_task2.json --output_dir ent_task2_hyp_th0.5_beamsearch/  --cache_dir ../../hfcache/ --n_search_beams 8  --hypothesis_acceptance_threshold 0.5  --search_algorithm beam_search --deductor_add_hyp


#PYTHONPATH=.. python proof_search.py --deductor_path ../outputs/deductor-flant5-xl/\
# --selector_path ../outputs/selector-flant5-xl/\
# --test_data_path ../data/test/custom_test.json\
# --output_dir .\
# --cache_dir ../../hfcache/\
# --n_search_beams 8\
# --hypothesis_acceptance_threshold 0.3\
# --deductor_add_hyp\
# --search_algorithm beam_search\
# --verbose

