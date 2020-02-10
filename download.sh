mkdir data
wget https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip -P data
unzip data/list_person_all_extended_utf8.zip
mv list_person_all_extended_utf8.csv data/list_person_all_extended_utf8.csv
python app/00_download.py 
python app/01_clean_text.py 