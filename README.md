# word2vec
Simple implementation for word2vec using TensorFlow 2.0

## setup 

```bash 
pyenv install 3.6.9
pyenv local 3.6.9 
python -m venv venv 
. venv/bin/activate 
pip install -r requirements.txt 
```

If you can use GPU, you can install requirements-gpu.txt.

```bash
pip install -r requirements-gpu.txt
```


## Prepare corpus 

```bash 
./download.sh
``` 

## Train Tokenizer

```bash 
python app/03_train_tokenizer.py
```

```bash 
python app/04_get_token_distribution.py
```

## Train model

```bash
python app/05_train.py
```
