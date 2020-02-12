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

## watch loss result  

Use tensorboard. 
Example. 

```bash 
tensorboard --logdir ./out/record/20200211-0111
```

## Evaluate vectors.  

Example.  

```bash 
 python app/06_eval.py ./out/record/20200211-1146/wordvec_009.npy  
 Enter the word
吾輩は猫である
Original word :
	Top 00 :  この
	Top 01 :  そして
	Top 02 :  ――
	Top 03 :  ......
	Top 04 :  ――
Original word :  吾
	Top 00 :  我が
	Top 01 :  我
	Top 02 :  われ
	Top 03 :  己
	Top 04 :  此
Original word :  輩
	Top 00 :  友
	Top 01 :  知りません
	Top 02 :  生きた
	Top 03 :  言
	Top 04 :  モノ
Original word :  は
	Top 00 :  はまた
	Top 01 :  はもう
	Top 02 :  は皆
	Top 03 :  はまだ
	Top 04 :  人は
Original word :  猫
	Top 00 :  洋服
	Top 01 :  犬
	Top 02 :  幽霊
	Top 03 :  女中
	Top 04 :  僕等
Original word :  である
	Top 00 :  です
	Top 01 :  だった
	Top 02 :  でした
	Top 03 :  なのである
	Top 04 :  となった
```

