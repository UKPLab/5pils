import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np


def cosine_similarity(vec1, vec2):
    '''
    Compute cosine similarity between two vectors.
    '''
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2)


def sort_with_clip_score(image_index, 
                        evidence_index_list,
                        image_embeddings, 
                        clip_evidence_embeddings):
    '''
    Sort the candidate evidence based on CLIP score.
    '''
    image = image_embeddings[image_index]
    similarities = []

    for idx in evidence_index_list:
        sim = cosine_similarity(image, clip_evidence_embeddings[idx]) 
        similarities.append((idx, sim))
    # Sort by similarity in descending order
    sorted_indices = sorted(similarities, key=lambda x: x[1], reverse=True)
    # Return only indices, not similarities
    return [idx for idx, _ in sorted_indices]


def sort_with_image_similarity(image_index,
                               train_images_index_list,
                               image_embeddings):
    '''
    Sort the candidate demonstrations based on CLIP similarity between images.
    '''
    image = image_embeddings[image_index]
    similarities = []
    for idx, emb_idx in train_images_index_list:
        sim = cosine_similarity(image, image_embeddings[emb_idx])
        similarities.append((idx, sim))
    # Sort by similarity in descending order
    sorted_indices = sorted(similarities, key=lambda x: x[1], reverse=True)
    # Return only indices, not similarities
    return [idx for idx, _ in sorted_indices]


def get_topk_evidence(image_path,
                      evidence,
                      image_embeddings,
                      clip_evidence_embeddings,
                      image_map,
                      k=3):
    '''
    Given an image, get the topk evidence using CLIP similarity.
    '''
    evidence = [ev for ev in evidence if ev['image path']==image_path]
    evidence_index = [evidence.index(ev) for ev in evidence if ev['image path']==image_path]
    if len(evidence_index)>k:
        image_index = int(image_map[image_path])
        sorted_evidence = sort_with_clip_score(image_index,
                                               evidence_index,
                                               image_embeddings,
                                               clip_evidence_embeddings)
        return sorted_evidence[:k]
    else:
        #If less than k evidence, skip ranking step and return all of them
        return evidence_index

    
def get_topk_demonstrations(image_path,
                            question,
                            train,
                            image_embeddings,
                            image_map,
                            k=2):
    '''
    Retrieve top k demonstrations from the train set based on image-image similarity with the test image.
    '''
    train_image_idx=[]
    #Only take train images that have a label for the corresponding task
    subset_train_index = [t for t in range(len(train)) if train[t][question].lower()!='not enough information']
    subset_train = [train[t] for t in range(len(train)) if t in subset_train_index]
    for idx, i in  zip(subset_train_index,[t['image path'] for t in subset_train]):
        try:
            train_image_idx.append((idx,int(image_map[i]))) #The candidate demonstrations index in the embedding matrix
        except:
            pass
    image_idx = int(image_map[image_path]) #The index of the test image in the embedding matrix
    sorted_candidates =  sort_with_image_similarity(image_idx,train_image_idx,image_embeddings)
    return sorted_candidates[:k]


def get_evidence_prompt(evidence):
    '''
    Given a set of evidence, generate the prompt.
    '''
    prompt = ''
    for ev in range(len(evidence)):
        text = 'Evidence %s\n'%ev
        if 'evidence url' in evidence[ev].keys():
            text += 'URL: %s\n'%evidence[ev]['evidence url']
        if 'hostname' in evidence[ev].keys():
            text += 'Hostname: %s\n'%evidence[ev]['hostname']
        if 'sitename'in evidence[ev].keys():
            text += 'Sitename: %s\n'%evidence[ev]['sitename']
        if 'title' in evidence[ev].keys():
            text += 'Title: %s\n'%evidence[ev]['title']
        if 'author' in evidence[ev].keys():
            text += 'Author: %s\n'%evidence[ev]['author']
        if 'date' in evidence[ev].keys():
            text += 'Date: %s\n'%evidence[ev]['date']
        if 'description' in evidence[ev].keys():
            text += '%s\n'%evidence[ev]['description']

        if 'Caption not found' not in evidence[ev]['image caption'] and 'Image not found' not in evidence[ev]['image caption']:
            text += 'Image captions: %s\n' % evidence[ev]['image caption']

        text += '\n'
        prompt += text
        
    return prompt


def truncate_text(evidence_text,tokenizer,max_length=128):
  '''
  Truncate the evidence text to the maximum token length accepted by the multilingual CLIP model.
  '''
  # Tokenize the input
  tokens = tokenizer.encode(evidence_text, add_special_tokens=True)
  if len(tokens) > max_length:
      # Truncate the tokens
      tokens = tokens[:max_length]
  # Convert tokens back to a string if needed
  return tokenizer.decode(tokens, skip_special_tokens=True)


def get_tokenized_evidence(evidence, tokenizer):
    '''
    Get tokenized to compute the text embeddings.
    '''
    text_list = []
    for s in tqdm(range(len(evidence))):
      text = ''
      text += evidence[s]['title']
      if 'Caption not found' not in evidence[s]['image caption'] and 'Image not found' not in evidence[s]['image caption']:
          text += evidence[s]['image caption']
      if text=='':
        text += evidence[s]['description']
      text = truncate_text(text,tokenizer)
      text_list.append(text)
    return text_list


def compute_clip_text_embeddings(texts, model, tokenizer,  batch_size=16):
    '''
    Compute the embeddings of evidence text passages
    '''
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            batch_embeddings = model.forward(batch_texts,tokenizer).detach()
            all_embeddings.append(batch_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def compute_clip_image_embeddings(image_paths, preprocess, model, batch_size=32):
    '''
    Compute image embeddings.
    '''
    all_embeddings = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in tqdm(range(0, len(image_paths), batch_size)):
        # Load and preprocess batch of images
        batch_images = [preprocess(Image.open(path)).unsqueeze(0) for path in image_paths[i:i+batch_size]]
        batch_images_tensor = torch.cat(batch_images).to(device)

        # Compute embeddings
        with torch.no_grad():
            batch_embeddings = model.encode_image(batch_images_tensor).detach()
            all_embeddings.append(batch_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)