import glob 
import re
from tqdm import tqdm 

text_paths = glob.glob("./data/text/*.txt")

regex_1 = re.compile("《.*?》|\［.*?\］")
def remove_rubi(lines):
    new_lines = [regex_1.sub("", line) for line in lines] 
    return new_lines

symbol = "--------------------------------------------"
def remove_explanation(lines):
    separate_symbol_cnt = 0
    new_lines = [] 
    for line in lines:
        if separate_symbol_cnt < 2 and  symbol in line:
            separate_symbol_cnt += 1
            continue 
        elif separate_symbol_cnt >= 2:
            new_lines.append(line)
    if separate_symbol_cnt == 0:
        return lines 
    return new_lines 

def remove_information(lines):
    # 後ろから文章を見ていって、改行記号のみの行が2つ見つかるまでの範囲は文書情報とみなして削除する
    n_cnt = 0
    idx = 0 
    for i, line in enumerate(lines[::-1]):
        if i == 0:
            continue
        elif line == "\n":
            n_cnt += 1
        if n_cnt >= 1:
            idx = i
            break 
    lines = lines[:-idx]
    return lines 

def remove_blank_lines(lines):
    new_lines = []
    for line in lines:
        if line != "\n":
            new_lines.append(line)
    return new_lines    

exception_cnt = 0
for text_path in tqdm(text_paths): 
    try:
        with open(text_path, "r", encoding='sjis') as f:
            lines =  f.readlines()
        lines = remove_rubi(lines)
        lines = remove_explanation(lines) # 文頭部分の説明文の削除
        lines = remove_information(lines) # 文末部分の説明文の削除
        lines = remove_blank_lines(lines) 
        with open("./data/corpus.txt", "a") as f:
            f.writelines(lines)
    except:
        exception_cnt += 1  
print(f"{exception_cnt:04d} exdeptions has occurred.")