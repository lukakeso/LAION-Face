
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
OLD_IMAGE_FOLDER = 'laion_face_data/glasses_imgs'
NEW_IMAGE_FOLDER = 'laion_face_data/selected_glasses_imgs'
os.makedirs(
    name=f'{cwd}/{NEW_IMAGE_FOLDER}',
    exist_ok=True
)


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

for file_name in tqdm(os.listdir(f'{cwd}/{OLD_IMAGE_FOLDER}')):
    # check only text files
    
    if file_name.endswith('.txt'):
        if not(Path(f"{NEW_IMAGE_FOLDER}/{file_name}").is_file()):
            file = open(f"{OLD_IMAGE_FOLDER}/{file_name}", 'r')
            text = file.read()
            file.close()
            text = replace_word_spacing_char(text).lower()
            if contains_noun_adjective(text, nouns = ["glasses", "sunglasses"]):
                new_file = open(f"{NEW_IMAGE_FOLDER}/{file_name}", 'w')
                new_file.write(text)
                new_file.close()
                
                

