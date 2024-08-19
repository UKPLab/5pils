import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from baseline.generation_utils import *
from evaluation.evaluation_metrics import *


if __name__=='__main__':
    clip_evidence_embeddings = np.load('dataset/embeddings/evidence_embeddings.npy')
    image_embeddings = np.load('dataset/embeddings/image_embeddings.npy')
    image_embeddings_map = load_json('dataset/embeddings/image_embeddings_map.json')
    evidence = load_json('dataset/retrieval_results/evidence.json')

    test = load_json('dataset/test.json')
    source = [t for t in test if t['source']!='not enough information']
    date = [t for t in test if t['date numeric label']!='not enough information']
    location = [t for t in test if t['location']!='not enough information']
    motivation = [t for t in test if t['motivation']!='not enough information']

    print('-----------')
    print('Time ranking')
    print('-----------')
    print('Source %s'%get_ndcg_score(source,'source',evidence,image_embeddings, clip_evidence_embeddings,image_embeddings_map,True))
    print('Date %s'%get_ndcg_score(source,'date numeric label',evidence,image_embeddings, clip_evidence_embeddings,image_embeddings_map,True))
    print('Location %s'%get_ndcg_score(source,'location',evidence,image_embeddings, clip_evidence_embeddings,image_embeddings_map,True))
    print('Motivation %s'%get_ndcg_score(source,'motivation',evidence,image_embeddings, clip_evidence_embeddings,image_embeddings_map,True))
    print('-----------')
    print('CLIP ranking')
    print('-----------')
    print('Source %s'%get_ndcg_score(source,'source',evidence,image_embeddings, clip_evidence_embeddings,image_embeddings_map,False))
    print('Date %s'%get_ndcg_score(source,'date numeric label',evidence,image_embeddings, clip_evidence_embeddings,image_embeddings_map,False))
    print('Location %s'%get_ndcg_score(source,'location',evidence,image_embeddings, clip_evidence_embeddings,image_embeddings_map,False))
    print('Motivation %s'%get_ndcg_score(source,'motivation',evidence,image_embeddings, clip_evidence_embeddings,image_embeddings_map,False))   
