
dl-tokenizer:
	huggingface-cli download meta-llama/Llama-3.1-8B  --local-dir Llama-3.1-8B

c4_train_10k:
	python download_c4_by_n.py -n 10000 

	PYTHONPATH=/workspace/NeMo:$$PYTHONPATH python -m scripts.nlp_language_modeling.preprocess_data_for_megatron \
		--dataset-impl mmap --workers 8  \
		--tokenizer-library huggingface \
		--tokenizer-type ./Llama-3.1-8B \
		--input c4_train_10000.jsonl \
		--output-prefix c4_train_10000

c4_val_10k:
	python download_c4_by_n.py -n 10000 --val

	PYTHONPATH=/workspace/NeMo:$$PYTHONPATH python -m scripts.nlp_language_modeling.preprocess_data_for_megatron \
		--dataset-impl mmap --workers 8  \
		--tokenizer-library huggingface \
		--tokenizer-type ./Llama-3.1-8B \
		--input c4_val_10000.jsonl \
		--output-prefix c4_val_10000