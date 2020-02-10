import sentencepiece as spm
sp = spm.SentencePieceProcessor()
spm.SentencePieceTrainer.Train("--input=./data/corpus.txt \
    --model_prefix=tokenizer/aozora_8k_model \
    --vocab_size=8000 --character_coverage=0.98 \
    --shuffle_input_sentence=true")