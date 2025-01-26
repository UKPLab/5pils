import os
import spacy
import argparse
import pandas as pd
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset_collection.preprocessing_utils import *
from evaluation.geonames_collection import *
from utils import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Preprocess the raw data to create a train, val, and test sets.')
    parser.add_argument('--json_file_path', type=str, default='dataset/gpt4_annotations/annotations.json',
                        help='Path to the GPT4 annotations')   
    parser.add_argument('--ner_model', type=str, default='en_core_web_lg',
                        help='The spacy model to extract dates NER tags.')
    parser.add_argument('--geonames_username', type=str, default=" ", 
                        #Insert here your GeoName username. Necessary for location
                        help='Username to access GeoNames API.')
    parser.add_argument('--sleep_geonames', type=int, default=2,
                        help='Waiting time between two API calls of the GeoNames API.')
    parser.add_argument('--geonames_data', type=str, default='dataset/geonames_results.json',
                        help='File to store the geonames results.')
    

    args = parser.parse_args()

    #Load spacy NLP model
    nlp = spacy.load(args.ner_model)

    raw_data = load_json(args.json_file_path)
    #Preprocess the data
    normalized_data = pd.DataFrame([normalize_json_fields(d , nlp) for d in raw_data])
    #Remove duplicate images
    duplicates = get_duplicates(normalized_data['image_path'].to_list())
    duplicates_mask = normalized_data['image_path'].apply(lambda row : False if row in duplicates else True)
    normalized_data = normalized_data[duplicates_mask]
    #Remove images with no labeled pillars
    annotation_count = normalized_data[['provenance','source','location', 'date', 'motivation']]
    for c in annotation_count.columns:
        annotation_count[c] = annotation_count[c].apply(lambda row : row if str(row).lower() != 'not enough information' else None)
    null_rows = annotation_count.isnull().all(axis=1)
    normalized_data = normalized_data.drop(normalized_data.index[null_rows])
    #Temporal ordering
    normalized_data = normalized_data.sort_values(by='publication_date')
    #Fill missing information and convert to list of dictionaries
    normalized_data = normalized_data.fillna('not enough information').to_dict(orient='records')
    #Get GeoNames entries for all ground truth locations
    all_locs = []
    for d in normalized_data:
        locations = [l for l in extract_named_entities(d['location'], nlp, 'locations')]
        for l in locations:
            if l not in all_locs:
                results = search_location(l,args.geonames_username,args.sleep_geonames)
                time.sleep(args.sleep_geonames)
                save_result(results,args.geonames_data)
                all_locs += locations
    #Train - Val - Test split
    train_idx = int(len(normalized_data) * 0.60)
    val_idx = int(len(normalized_data) * 0.70)

    # Perform the split
    train = normalized_data[:train_idx]
    val = normalized_data[train_idx:val_idx]
    test = normalized_data[val_idx:]

    # Save files
    with open('dataset/train_custom.json', 'w') as file:
        json.dump(train, file, indent=4)
    with open('dataset/val_custom.json', 'w') as file:
        json.dump(val, file, indent=4)
    with open('dataset/test_custom.json', 'w') as file:
        json.dump(test, file, indent=4)
