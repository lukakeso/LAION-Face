import torch
import pyarrow.parquet as pq
import pyarrow as pa
import os
from tqdm import tqdm
import argparse
import pandas as pd

import spacy
from spacy.matcher import DependencyMatcher
nlp = spacy.load("en_core_web_sm")

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
    
    contains_noun_adj = []
    
    for text in tqdm(all_texts):        
        if contains_noun_adjective(text, nouns = ["glasses", "sunglasses", "frames", "goggles", "shades", "spectacles", "eyewear", "eye-wear"]):
            contains_noun_adj.append(True)
        else:
            contains_noun_adj.append(False)
            
    return contains_noun_adj


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--laion_face_meta_dir", default="laion_face_meta", help="directory with the .parquet files")
    parser.add_argument("--filter_word", default="glasses_synonims", help="word by which the TEXT column will be filtered by")
    parser.add_argument("--no_filter_adj", action='store_false', help="do additional filtering with adjectives")
    parser.add_argument("--no_nums", action='store_false', help="do additional filtering with numbers")
    parser.add_argument("--no_filter_adj_pos", action='store_false', help="do additional filtering with adjective position")

    
    args=parser.parse_args()
    glasses_syn = None
    if args.filter_word == "glasses_synonims":
        glasses_syn = ["glasses", "frames", "goggles", "shades", "spectacles", "eyewear", "eye-wear"]
    #200 glasses adjectives
    glasses_adj = ['absurd', 'aerodynamic', 'amber', 'ancient', 'antique', 'beige', 'big', 'black', 'blind', 'blue', 
                   'brazilian', 'bright', 'broad', 'brown', 'browline', 'built-in', 'cheap', 'chromed', 'chunky', 
                   'circular', 'classic', 'color', 'cool', 'crooked', 'danish', 'dark', 'dim', 'dirty', 'disposable', 
                   'dumb', 'dusty', 'eccentric', 'enormous', 'exclusive', 'expensive', 'extra', 'extremely', 'familiar', 
                   'fancy', 'flowered', 'full', 'fun', 'generic', 'gray', 'great', 'green', 'grey', 'handmade', 'hard', 
                   'heav', 'hot', 'huge', 'fitting', 'door', 'interesting', 'italian', 'large', 'lavender', 'lilac', 
                   'little', 'masculine', 'metal', 'mexican', 'modern', 'narrow', 'neat', 'new', 'normal', 'opaque', 
                   'orange', 'ordinary', 'outrageous', 'oval', 'oversize', 'pale', 'perfect', 'pink', 'precious', 'purple', 
                   'rectangular', 'red', 'reflective', 'ridiculous', 'rimless', 'rimmed', 'sensitive', 'serious', 'short', 
                   'sleek', 'slick', 'slim', 'slippery', 'small', 'smart', 'smoky', 'spanish', 'sparkl', 'special', 'sport', 
                   'standard', 'stupid', 'stylish', 'tactical', 'tasteful', 'terrible', 'thick', 'tiny', 'top', 'trendy',
                   'triangular', 'usual', 'weird', 'white', 'wide', 'wonderful', 'wooden', 'yellow', 'square', 'triangle',
                   'round', 'cat', 'thin', 'bottom', 'curv', 'straight', 'horn', 'old', 'octagonal', 'elegant', 'clear', 
                   'transparent', 'gold', 'silver', 'striped', 'turquoise', 'violet', 'bronze', 'lime', 'indigo', 'cyan', 
                   'maroon', 'magenta', 'teal', 'shin', 'crystal', 'shaped', 'scarlet', 'adidas', 'boss', 'gucci', 'bvlgari',
                   'bottega', 'balenciaga', 'dior', 'calvin', 'dkny', 'armani', 'guess', 'kors', 'moschino', 'nike', 'oakley', 
                   'lauren', 'vogue', 'versace', 'burberry', 'dolce', 'diesel', 'lacoste', 'tom', 'tight', 'plastic', 'ray',
                   'prescription', 'near', 'far', 'heart', 'festiv', 'rav', 'party', 'wayfarer', 'oblong', 'diamond', 'safety',
                   'lab', 'reading', 'protect', 'light', 'fake', 'aviator', 'wrap', 'geometric', 'pearl', 'polish', 'bike',
                   'ski', 'designer', 'camo', 'natural', 'medium'] 

    meta_dir = os.listdir(args.laion_face_meta_dir)

    if not(f'laion_face_{args.filter_word}.parquet' in meta_dir):
        all_parquet_files = [f for f in meta_dir if "laion_face_part_000" in f] #original parquet files
        big_table_filtered = pd.DataFrame()
        for parquet_file in tqdm(all_parquet_files):
            big_table=pq.read_table(os.path.join(args.laion_face_meta_dir, parquet_file)).to_pandas()
            big_table["TEXT"] = big_table["TEXT"].str.lower()
            big_table["TEXT"] = big_table["TEXT"].str.replace('_',' ')
            big_table["TEXT"] = big_table["TEXT"].str.replace('-',' ')
            big_table["TEXT"] = big_table["TEXT"].str.replace('+',' ')
            
            if args.filter_word == "glasses_synonims":
                for syn in glasses_syn:
                    mask = big_table["TEXT"].str.contains(syn, case=False, na=False)
                    masked_table = big_table.loc[mask]                                  
                    big_table_filtered = pd.concat([big_table_filtered, masked_table])      
            else:
                mask = big_table["TEXT"].str.contains(args.filter_word, case=False, na=False)   #binary mask for all rows that contain the filtered word
                masked_table = big_table.loc[mask]                                              #contains rows with filtered word in TEXT column
                big_table_filtered = pd.concat([big_table_filtered, masked_table])              #aggregate of all the rows
            
        
        print("before droping duplicates:", len(big_table_filtered.index))
        big_table_filtered = big_table_filtered.drop_duplicates()
        nr_filtered = len(big_table_filtered.index)
        print(f"DONE!")
        print(f"Filtering with word '{args.filter_word}' yielded {nr_filtered} entries.")
        print(f"SAVING DATA...")
        pq.write_table(pa.Table.from_pandas(big_table_filtered), os.path.join(args.laion_face_meta_dir,f'laion_face_{args.filter_word}.parquet'))
        print(f"--------------------------------------------------")

    else:
        print("TABLE ALREADY EXISTS, READING FROM FILE")
        big_table_filtered = pq.read_table(os.path.join(args.laion_face_meta_dir, f'laion_face_{args.filter_word}.parquet')).to_pandas()
        print(f"--------------------------------------------------")
    
    if args.no_nums and not(f'laion_face_{args.filter_word}_nums.parquet' in meta_dir):
        print("NOW FILTERING FOR NUMBERS")
        nums_mask = big_table_filtered["TEXT"].str.contains('.*[0-9].*', case=False, na=True, regex=True)==False
        big_table_filtered_nums = big_table_filtered.loc[nums_mask]
        
        nr_filtered = len(big_table_filtered_nums.index)
        print(f"DONE!")
        print(f"Filtering numbers yielded {nr_filtered} entries.")
        print(f"SAVING DATA...")
        pq.write_table(pa.Table.from_pandas(big_table_filtered_nums), os.path.join(args.laion_face_meta_dir,f'laion_face_{args.filter_word}_nums.parquet'))
        print(f"--------------------------------------------------")
        
    
    if args.no_filter_adj and not(f'laion_face_{args.filter_word}_adjectives.parquet' in meta_dir):
        big_table_filtered_adj = pd.DataFrame()
        print("NOW FILTERING FOR ADJECTIVES")
        for adj in tqdm(glasses_adj):
            mask = big_table_filtered["TEXT"].str.contains(adj, case=False, na=False)
            masked_table = big_table_filtered.loc[mask]                                  
            big_table_filtered_adj = pd.concat([big_table_filtered_adj, masked_table])    

        print("before droping duplicates:", len(big_table_filtered_adj.index))
        big_table_filtered_adj = big_table_filtered_adj.drop_duplicates()
        nr_filtered = len(big_table_filtered_adj.index)
        print(f"DONE!")
        print(f"Filtering with adjectives yielded {nr_filtered} entries.")
        print(f"SAVING DATA...")
        pq.write_table(pa.Table.from_pandas(big_table_filtered_adj), os.path.join(args.laion_face_meta_dir,f'laion_face_{args.filter_word}_adjectives.parquet'))
        print(f"--------------------------------------------------")

    if args.no_filter_adj_pos and not(f'laion_face_{args.filter_word}_adjpos.parquet' in meta_dir):
        print("NOW FILTERING FOR ADJECTIVE POSITION")
        all_text = big_table_filtered["TEXT"].to_list()
        adjective_pos = filter_by_adjective(all_text)
        big_table_filtered["adjpos"] = adjective_pos
        big_table_filtered_adj_pos = big_table_filtered[big_table_filtered["adjpos"]]  
        big_table_filtered_adj_pos.drop(columns=['adjpos']) 

        print("before droping duplicates:", len(big_table_filtered_adj_pos.index))
        big_table_filtered_adj_pos = big_table_filtered_adj_pos.drop_duplicates()
        nr_filtered = len(big_table_filtered_adj_pos.index)
        print(f"DONE!")
        print(f"Filtering with adjective position yielded {nr_filtered} entries.")
        print(f"SAVING DATA...")
        pq.write_table(pa.Table.from_pandas(big_table_filtered_adj_pos), os.path.join(args.laion_face_meta_dir,f'laion_face_{args.filter_word}_adjpos.parquet'))
        print(f"--------------------------------------------------")
    