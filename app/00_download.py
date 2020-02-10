import pandas as pd 
import os 
import subprocess as sub 
from tqdm import tqdm
from joblib import Parallel, delayed

df = pd.read_csv("./data/list_person_all_extended_utf8.csv")
df = df[df["作品著作権フラグ"] == "なし"] # 著作権のない作者だけ抜き出す
urls = df['テキストファイルURL']


os.makedirs("./data/zip", exist_ok=True)
os.makedirs("./data/text", exist_ok=True)

def download_zip(url):
    devnull = open('/dev/null', 'w')
    cmd = ["wget", url, "-P", "data/zip"] 
    sub.Popen(cmd, stdout=devnull, stderr=devnull)
    save_path = os.path.join('data/zip/', url.split('/')[-1])
    unzip_cmd = ["unzip", save_path, "-d", "./data/text"]
    sub.Popen(unzip_cmd, stdout=devnull, stderr=devnull)
    devnull.close() 

results = Parallel(n_jobs=5, verbose=10)(delayed(download_zip)(str(url)) for url in urls)
