import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from evaluation.evaluation_metrics import *
from utils import *
import argparse
import spacy

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate generated answers')
    parser.add_argument('--results_file', type=str, default='output/results_date.json',
                        help='Path to the predicted answers.')
    parser.add_argument('--task', type=str, default='date',
                        help='Which question to evaluate. One of [source, date, location, motivation]')
    parser.add_argument('--ner_model', type=str, default='en_core_web_lg',
                        help='The spacy model to extract dates NER tags.')
    parser.add_argument('--geonames_username', type=str, default=" ", #Insert here your GeoName username. Necessary for location only
                        help='Username to access GeoNames API.')
    parser.add_argument('--sleep_geonames', type=int, default=2,
                        help='Waiting time between two API calls of the GeoNames API.')
    parser.add_argument('--geonames_data', type=str, default='dataset/geonames_results.json',
                        help='File to store the geonames results.')

    args = parser.parse_args()

    results = load_json(args.results_file)
    rougeL = []  
    meteor = []
    exact_match = []
    delta = []
    codelta = []
    hldelta = []
    berts  = []
    nlp = spacy.load(args.ner_model)


    for r in results:
        if args.task ==  'source':
            scores = evaluate(r['output'], r['ground_truth'], args.task)
            rougeL.append(scores['rougeL'])
            meteor.append(scores['meteor'])
        elif args.task=='date':
            scores = evaluate(r['output'], r['ground_truth'], args.task, nlp)
            exact_match.append(scores['exact_match'])
            delta.append(scores['delta'])
        elif args.task=='location':

            geonames_entries = set([d['query'] for d in load_json(args.geonames_data)])
            NER_ground_truth = [e for e in extract_named_entities(r['ground_truth'], nlp, 'locations')  if e in geonames_entries]
            scores = evaluate(r['output'], r['ground_truth'], args.task)
            if len(NER_ground_truth) > 0:
                #Compute the location delta metrics too because the ground truth has Geonames entries
                NER_scores = evaluate(r['output'], r['ground_truth'], 'location NER', nlp, args.geonames_data, args.geonames_username, args.sleep_geonames)
                #codelta and hldelta is only reported for the subset with GeoNames entries
                codelta.append(NER_scores['codelta'])
                hldelta.append(NER_scores['hldelta'])
                if hldelta ==1:
                    #If there is an exact match in terms of GeoNames entries, i.e., HLDelta is equal to 1, then we count the prediction has accurate.
                    scores['meteor']=1
                    scores['rougeL']=1


            rougeL.append(scores['rougeL'])
            meteor.append(scores['meteor'])
        elif args.task ==  'motivation':
            scores = evaluate(r['output'], r['ground_truth'], args.task)
            rougeL.append(scores['rougeL'])
            meteor.append(scores['meteor'])
            berts.append(scores['BertS'])
        else:
            print('Invalid task name')
            break

    print('------------------------')
    print('Evaluation for task: ' + args.task)
    print('------------------------')
    if args.task=='source':
        print('RougeL score %s'%np.mean(rougeL))
        print('Meteor score %s'%np.mean(meteor))
    elif args.task=='date':
        print('EM score %s'%np.mean(exact_match))
        print('Delta score %s'%np.mean(delta))
    elif args.task=='location':
        print('RougeL score %s'%np.mean(rougeL))
        print('Meteor score %s'%np.mean(meteor))
        print('OCDelta score %s'%np.mean(codelta))
        print('HL Delta score %s'%np.mean(hldelta))
    elif args.task=='motivation':
        print('RougeL score %s'%np.mean(rougeL))
        print('Meteor score %s'%np.mean(meteor))
        print('Bert score %s'%np.mean(berts))
    else:
        print('Invalid task name')
