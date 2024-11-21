import torch
import clip
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer
import numpy as np
import os
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from baseline.generation_utils import *


if __name__=='__main__':
    if not 'embeddings' in os.listdir('dataset/'):
        os.mkdir('dataset/embeddings')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
    # Load Model & Tokenizer
    text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #Evidence
    evidence = load_json('dataset/retrieval_results/evidence.json')
    text_list = get_tokenized_evidence(evidence,tokenizer)
    evidence_embeddings = compute_clip_text_embeddings(text_list, text_model, tokenizer, batch_size=16)
    np.save('dataset/embeddings/evidence_embeddings.npy', evidence_embeddings)

    #Images: used for demonstration selection based on image similarity
    image_model, preprocess = clip.load('ViT-L/14', device=device)
    image_paths = ['dataset/processed_img/' + i for i in os.listdir('dataset/processed_img/')]
    # map each image to its index in the embedding matrix
    list_dict = {image_paths[i]: str(i) for i in range(len(image_paths))}
    # Save the dictionary to a JSON file
    with open('dataset/embeddings/image_embeddings_map.json', 'w') as json_file:
        json.dump(list_dict, json_file)
    image_embeddings = compute_clip_image_embeddings(image_paths, preprocess, image_model)
    np.save('dataset/embeddings/image_embeddings.npy', image_embeddings)