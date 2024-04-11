
import os
import pyarrow.parquet as pq
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize

import spacy
from spacy.matcher import DependencyMatcher
nlp = spacy.load("en_core_web_sm")


cwd = os.getcwd()
IMAGE_FOLDER = 'laion_face_data/glasses_imgs'
os.makedirs(
    name=f'{cwd}/{IMAGE_FOLDER}',
    exist_ok=True
)
NEW_IMAGE_FOLDER = 'laion_face_data/selected_glasses_imgs'
os.makedirs(
    name=f'{cwd}/{NEW_IMAGE_FOLDER}',
    exist_ok=True
)

def wget_downloader(data):
    for img_name, (link, _) in tqdm(data.items()):
        try:
            download_path = f'{IMAGE_FOLDER}/{img_name}'
            #print(f'Downloading: {link}')
            command = ["wget", link, "-O", download_path, "-q", "--waitretry=1", "--timeout=15"]
            subprocess.run(command, timeout=10)  #--show-progress
        except subprocess.TimeoutExpired as te:
            print(te)
        #p = subprocess.Popen(command, stdout=subprocess.PIPE)
        #stdout, stderr = p.communicate()

        #os.system(f'wget {link} -O {img_name}')
        #print(".")
            # with open(img_name,'wb') as f:
            #     shutil.copyfileobj(res.raw, f)
            #     print('Image sucessfully Downloaded: ',img_name)

def filter_downloaded(data):
    already_downloaded_imgs = os.listdir(IMAGE_FOLDER)
    for i in already_downloaded_imgs:
        if i in data:
            idx = i.split(".")[0]
            if not(Path(f"{IMAGE_FOLDER}/{idx}.txt").is_file()):
                file = open(f"{IMAGE_FOLDER}/{idx}.txt", 'w')
                file.write(data[i][1])
                file.close()
            data.pop(i)
    return data

def replace_word_spacing_char(text):

    c0 = text.count("-")
    c1 = text.count("_")
    c2 = text.count("+")
    
    if c0 > c1 and c0 > c2:
        return text.replace("-", " ")
    elif c1 > c0 and c1 > c2:
        return text.replace("_", " ")
    else:
        return text.replace("+", " ")

def contains_noun_adjective(sentence, nouns):
    pattern = [
    {
        "RIGHT_ID": "target",
        "RIGHT_ATTRS": {"POS": "NOUN"}
    },
    # founded -> subject
    {
        "LEFT_ID": "target",
        "REL_OP": ">",
        "RIGHT_ID": "modifier",
        "RIGHT_ATTRS": {"DEP": {"IN": ["amod", "nummod"]}}
    },
    ]
    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("FOUNDED", [pattern])
        
    doc = nlp(sentence)
    for match_id, (target, modifier) in matcher(doc):
        if str(doc[target]) in nouns:
            #print(doc[modifier], doc[target], sep="\t")
            return True
            
    return False

def filter_by_adjective(all_texts):
    
    filtered_texts = []
    
    for text in tqdm(all_texts):        
        if contains_noun_adjective(text, nouns = ["glasses", "sunglasses"]):
            filtered_texts.append(text)
            
    return filtered_texts

parser=argparse.ArgumentParser()
parser.add_argument("--laion_face_meta_dir", default="laion_face_meta", help="directory with the .parquet files")
parser.add_argument("--filter_word", default="glasses_synonims_nums_adjectives", help="word by which the TEXT column will be filtered by")
args=parser.parse_args()

parquet_file = f'laion_face_{args.filter_word}.parquet'

big_table=pq.read_table(os.path.join(args.laion_face_meta_dir, parquet_file)).to_pandas()

all_links_ = big_table["URL"].to_list()
all_text_ = big_table["TEXT"].to_list()
all_text_ = [t.lower() for t in all_text_]

# replaces word spacers like "-", "_", "+" with space to be used by the tokenizer
for i, t in enumerate(all_text_):
    if not " " in t:
        all_text_[i] = replace_word_spacing_char(t)

adjective_text = filter_by_adjective(all_text_)

print(len(all_text_))
print(len(adjective_text))
quit()
all_links_w_text = {f'{i}'.zfill(6)+'.png': [l, t] for i, (l, t) in enumerate(zip(all_links_, all_text_))}

nr = len(all_links_w_text)
while len(all_links_w_text) > 0:
    all_links_w_text = filter_downloaded(all_links_w_text)
    print(f'***** Need to download {len(all_links_w_text)}/{nr} images *****')
    wget_downloader(all_links_w_text)