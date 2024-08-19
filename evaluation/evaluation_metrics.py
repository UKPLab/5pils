from evaluate import load
from dateutil import parser
import numpy as np
from scipy.optimize import linear_sum_assignment
from dateutil.tz import tzutc
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import ast
from dateutil.relativedelta import relativedelta
from haversine import haversine, Unit
from itertools import combinations
from utils import *
from dataset_collection.preprocessing_utils import *
from evaluation.geonames_collection import *


#Load the metrics
meteor = load('meteor')
rouge = load('rouge')
bertscore = load("bertscore")


def convert_to_date(date_str):
    '''
    Convert a string to a Date object
    '''
    try:
        return parser.parse(date_str)
    except (ValueError, TypeError):
        return None


def date_distance(dt1, dt2):
    '''
    Compute the distance between two dates
    '''
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=tzutc())
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=tzutc())
    if dt1 is None or dt2 is None:
        return float('inf')

    delta = relativedelta(dt1, dt2)
    return abs(delta.years + delta.months / 12 + delta.days / 365.25)


def location_coordinate_distance(coordinates1,coordinates2,unit=1000):
    '''
    Compute the coordinates distance between the prediction and ground truth. 
    Compare all pairs of GeoNames entities and take the smallest distance as optimist heuristic.
    '''
    d = min([haversine(c1,c2,unit=Unit.KILOMETERS) for c1 in coordinates1 for c2 in coordinates2])
    d /= unit
    return d


def hierarchical_distance_metric(pred_hierarchy, gt_hierarchy):
    '''
    Compute the distance between two hierarchies based on their common parent.
    '''
    if  all(i in pred_hierarchy for i in gt_hierarchy):
        return 0
    else:
        common_length = 0
        for p, g in zip(pred_hierarchy, gt_hierarchy):
            if p == g:
                common_length += 1
            else:
                break
        return len(pred_hierarchy) + len(gt_hierarchy) - 2 * common_length


def location_hierarchy_distance(hierarchy1,hierarchy2):
    '''
    Compute the distance between two locations given their GeoNames hierarchies.
    Compare all pairs of GeoNames entities and take the smallest distance as optimist heuristic.
    '''
    d = min([hierarchical_distance_metric(h1,h2) for h1 in hierarchy1 for h2 in hierarchy2])
    return d


def is_strict_subset(sublist, mainlist):
    '''
    Compute whether the content of one list is the subset of the content of another list.
    '''
    return set(sublist).issubset(set(mainlist)) and len(sublist) < len(mainlist)


def find_locations_to_remove(l):
    '''
    Remove locations for which the hierarchy is a strict subset of another location.
    '''
    indices_to_remove = []
    
    def contains_strict_subset(outer_list, other_lists):
        for sublist in outer_list:
            for other_list in other_lists:
                for other_sublist in other_list:
                    if is_strict_subset(sublist, other_sublist):
                        return True
        return False

    for i, outer_list in enumerate(l):
        if contains_strict_subset(outer_list, [other_list for j, other_list in enumerate(l) if i != j]):
            indices_to_remove.append(i)
    
    return indices_to_remove


def evaluate(prediction, 
             ground_truth, 
             task, 
             NER_model = None,
             geonames_data_path=None,
             geonames_username=None,
             sleep_geonames=2):
    '''
    Main 5Pils evaluation function.
    Params:
        prediction (str): a string containing the prediction for an image
        ground_truth (str): the ground truth context information
        NER_model (object): a spacy NER model to extract dates and location named entities from predictions.
        geonames_data_path (str): path to the json file storing the geoname entries. Only needed for the location task
        geonames_username (str): user name to connect to the GeoNames API. Only needed for the location task
        sleep_geonames (int): the waiting time in seconds between two calls of the GeoNames API
    '''
    #Source
    if task=="source":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = meteor.compute(predictions=prediction, references=[ground_truth])['meteor']
        return {'rougeL':rouge_result,"meteor": meteor_result}

    #Motivation
    elif task=="motivation":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = meteor.compute(predictions=prediction, references=[ground_truth])['meteor']
        berts_result = bertscore.compute(predictions=prediction,references=ground_truth,
                                        lang='en', model_type="distilbert-base-uncased")['f1'][0]
        return {'rougeL':rouge_result,"meteor": meteor_result, 'BertS':berts_result}
    
    #Location
    elif task=="location":
        if not isinstance(prediction, list):
            prediction = [prediction]
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        rouge_result = rouge.compute(predictions=prediction, references=[ground_truth])['rougeL']
        meteor_result = meteor.compute(predictions=prediction, references=[ground_truth])['meteor']
        return {'rougeL':rouge_result,"meteor": meteor_result}
    
    elif task == "location NER":
            geonames_data = load_json(geonames_data_path)
            geonames_entries = list(set([d['query'].lower() for d in geonames_data]))
            prediction_location_NER = [l for l in extract_named_entities(prediction,NER_model,'locations')]
            prediction_coordinates = []
            prediction_hierarchies = []
            matching_records = []
            #Prepare the predictions
            for p in prediction_location_NER:
                if p.lower() not in geonames_entries: 
                        #Add a new entry to the collected GeoName database if the prediction is not there yet
                        matching_records = search_location(p,geonames_username,sleep_geonames)
                        time.sleep(sleep_geonames)
                        save_result(matching_records,geonames_data_path)
                else:
                    matching_records = [d for d in geonames_data if 'coordinates' in d.keys() and d['query'].lower()==p.lower()]
                if len(matching_records) > 0 :            
                    prediction_coordinates.append([r['coordinates'] for r in matching_records])
                    prediction_hierarchies.append([r['hierarchy'] for r in matching_records])
            ground_truth_location_NER = [l for l in extract_named_entities(ground_truth,NER_model,'locations')] 
            ground_truth_coordinates = []
            ground_truth_hierarchies = []
            for g in ground_truth_location_NER:
                matching_records = [d for d in geonames_data if 'coordinates' in d.keys() and d['query'].lower()==g.lower()]
                if len(matching_records) > 0 : 
                    ground_truth_hierarchies.append([r['hierarchy'] for r in matching_records])
                    ground_truth_coordinates.append([r['coordinates'] for r in matching_records])
            idx_to_remove  = find_locations_to_remove(ground_truth_hierarchies)
            ground_truth_coordinates = [ground_truth_coordinates[i] for i in range(len(ground_truth_coordinates)) if i not in idx_to_remove]
            ground_truth_hierarchies = [ground_truth_hierarchies[i] for i in range(len(ground_truth_hierarchies)) if i not in idx_to_remove]
            if len(prediction_coordinates) > 0:
                
                if len(prediction_coordinates) > len(ground_truth_coordinates):
                    # Generate all combinations of size up to x
                    candidates = []
                    size = len(ground_truth_coordinates)
                    candidates.extend(combinations(prediction_coordinates, size))
                else: 
                    candidates = [prediction_coordinates]

                best_codelta = 0
                for candidate in candidates:
                #We find the minimal distance among all pairs
                    distances = np.array([[location_coordinate_distance(pc, gc) for gc in ground_truth_coordinates] for pc in candidate])           
                    row_ind, col_ind = linear_sum_assignment(distances)
                    scores = 0
                    non_zero_distance_list = [distances[r,c] for r, c in zip(row_ind, col_ind)]
                    non_zero_distance_list = sorted(non_zero_distance_list)
                    for d in non_zero_distance_list:
                        scores += 1/(1+d)
                    
                    coefficient = 1/len(ground_truth_coordinates) 
                    codelta = coefficient *scores
                    if codelta > best_codelta:
                        best_codelta = codelta
                
            else:
                best_codelta = 0

            if len(prediction_hierarchies) > 0:
                
                if len(prediction_hierarchies) > len(ground_truth_hierarchies):
                    # Generate all combinations of size up to x
                    candidates = []
                    size = len(ground_truth_hierarchies)
                    candidates.extend(combinations(prediction_hierarchies, size))
                else: 
                    candidates = [prediction_hierarchies]

                best_hierarchy_delta = 0
                for candidate in candidates:
                    distances = np.array([[location_hierarchy_distance(pc, gc) for gc in ground_truth_hierarchies] for pc in candidate])          
                    row_ind, col_ind = linear_sum_assignment(distances)
                    scores = 0
                    non_zero_distance_list = [distances[r,c] for r, c in zip(row_ind, col_ind)]
                    non_zero_distance_list = sorted(non_zero_distance_list)
                    for d in non_zero_distance_list:
                        scores += 1/(1+d)
                    coefficient = 1/len(ground_truth_hierarchies) 
                    hierarchy_delta  = coefficient *scores
                    if hierarchy_delta > best_hierarchy_delta :
                        best_hierarchy_delta = hierarchy_delta 
            else:
                best_hierarchy_delta  = 0
            return {"codelta": best_codelta, "hldelta": best_hierarchy_delta}
    #Date
    elif task == "date":
        if prediction!='':
            if prediction[0]=='[':
                prediction = prediction[1:-1]
        prediction_dates = extract_named_entities(prediction, NER_model,'dates_and_times')
        prediction_dates = [convert_to_date(date_str) for date_str in prediction_dates]
        prediction_dates = [d for d in prediction_dates if d is not None]
        ground_truth_dates = [convert_to_date(date_str) for date_str in ast.literal_eval(ground_truth)]
        if len(ground_truth_dates) > 0 and len(prediction_dates) > 0:
            if len(prediction_dates) > len(ground_truth_dates):
                # Generate all combinations of size up to x
                candidates = []
                size = len(ground_truth_dates)
                candidates.extend(combinations(prediction_dates, size))
            else: 
                candidates = [prediction_dates]
            best_delta = 0
            best_EM = 0
            for candidate in candidates:
                distances = np.array([[date_distance(pd, gd) for gd in ground_truth_dates] for pd in candidate])          
                row_ind, col_ind = linear_sum_assignment(distances)
                scores = 0
                non_zero_distance_list = [distances[r,c] for r, c in zip(row_ind, col_ind)]
                non_zero_distance_list = sorted(non_zero_distance_list)
                for d in non_zero_distance_list:
                    scores += 1/(1+d)
                exact_match = np.all(distances[row_ind, col_ind] == 0)
                coefficient = 1/len(ground_truth_dates) 
                delta = coefficient *scores
                if delta > best_delta:
                    best_delta = delta
                    best_EM = exact_match
            
            if len(prediction_dates) > len(ground_truth_dates):
                best_EM=0

            return {"exact_match": best_EM, "delta": best_delta}
        else:
            return {"exact_match": 0, "delta": 0}   
    else:
        raise ValueError("Invalid task name")
    


####################################
# Metrics for the evidence ranking #
####################################

def cosine_similarity(vec1, vec2):
    '''
    Compute cosine similarity between two embeddings.
    '''
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2)


def compute_clip_score(image_index, 
                       evidence_index_list,
                       image_embeddings,
                       clip_evidence_embeddings):
    '''
    Compute the similarity between an image and a list of textual evidence
    Params:
        image_index (int): the index of the image in the image embedding matrix.
        evidence_index_list (list): a list containing the indexes of the evidence in the evidence embedding matrix.
        image_embeddings (numpy.array): a matrix containing image embeddings (computed with CLIP)
        clip_evidence_embeddings (numpy.array): a matrix containing evidence embeddings (computed with CLIP)
    '''
    #Get the image embedding
    image = image_embeddings[image_index]
    similarities = []
    #Loop over the evidence
    for idx in evidence_index_list:
        sim = cosine_similarity(image, clip_evidence_embeddings[idx])
        similarities.append((idx, sim))
    return [score for _, score in similarities]


def generate_ngrams(text, n=3):
    '''
    Generate n-grams from a given text 
    '''
    try:
        vectorizer = CountVectorizer(ngram_range=(1,n),stop_words='english')
        vectorizer.fit_transform([text])
        return set(vectorizer.get_feature_names_out())
    except:
        return None

def ngram_overlap_score(passage, answer, n=2):
    '''
     Compute n-gram overlap score
    '''
    passage_ngrams = generate_ngrams(passage, n)
    answer_ngrams = generate_ngrams(answer, n)
    if answer_ngrams and passage_ngrams:
        overlap = passage_ngrams.intersection(answer_ngrams)
        return len(overlap) / max(len(passage_ngrams), len(answer_ngrams), 1)
    else:
        return 0


def eval_ranking(evidence,answer,predicted_ranking):
    '''
    Compute the NDCG score between a predicted ranking and the target ranking.
    '''
    ngram_scores = [ngram_overlap_score(e, answer) for e in evidence] 
    target_ranking = np.array(ngram_scores)
    if len(target_ranking)>1 and len(target_ranking)==len(predicted_ranking):
        ndcg = ndcg_score([target_ranking], [predicted_ranking])
        return ndcg
    else:
        return None
    

def get_ndcg_score(dataset, task, evidence, image_embeddings, clip_evidence_embeddings, image_embeddings_map,sort_with_date=False):
    '''
    Reports the NDCG for a specifc ranking method on a specific task.
    '''
    total_ndcg = 0
    count = 0
    img_corpus = [image['image path'] for image in dataset]
    ground_truth = [image[task] for image in dataset]
    for i in range(len(img_corpus)):
        evidence_subset = [ev for ev in evidence if ev['image path']==img_corpus[i]]
        evidence_subset_index = [evidence.index(ev) for ev in evidence if ev['image path']==img_corpus[i]]
        if len(evidence_subset)>3:
            #Retrieve the index of the image in the embedding matrix
            image_index = int(image_embeddings_map[img_corpus[i]])
            if sort_with_date:
                date_sort = pd.DataFrame(evidence_subset).reset_index().sort_values(by='date',ascending=False).index.to_list()
                predicted_ranking = [date_sort.index(i) for i in pd.DataFrame(evidence_subset).reset_index().index.to_list()]
            else:   
                predicted_ranking  = compute_clip_score(image_index,evidence_subset_index,image_embeddings, clip_evidence_embeddings)
            evidence_text = [ text[2:] for text in get_evidence_prompt(evidence_subset).split('Evidence ')[1:]]
                #only evaluate ranking when there are more than 3 evidence to select
            ndcg = eval_ranking(evidence_text,ground_truth[i],predicted_ranking)
            if ndcg !=None:
                total_ndcg +=ndcg
                count+=1
    if count != 0:
        return round(100*total_ndcg/count,2)
    else: 
        return 'No matching evidence for those images'
