import argparse
import os
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Heuristic to identify the unaltered, original image among the RIS results.')
    parser.add_argument('--json_path', type=str, default='dataset/manipulation_detection_test.json',
                        help='Path to the manipulation detection predictions')
    parser.add_argument('--download_image', type=int, default=0,
                        help='If True, download the images retrieved by RIS for images predicted as manipulated.')
    parser.add_argument('--map_json_path', type=str, default='dataset/map_manipulated_original.json',
                        help='Path to store the file that maps manipulated images to their identified original version.')
    args = parser.parse_args()

    if 'manipulated_original_img' not in os.listdir('dataset/'):
        os.mkdir('dataset/manipulated_original_img/')


    test = load_json('dataset/test.json')
    evidence = load_json('dataset/retrieval_results/evidence.json')

    manipulation_detection_test_image_paths = [im['image path'] for im in load_json(args.json_path) if im['manipulation detection']=='manipulated']
    if args.download_image:
        #Load data
        subset_evidence = [ev['image url']  for ev in evidence if ev['image path'] in manipulation_detection_test_image_paths]
        subset_evidence = [url if url else '' for url in subset_evidence]
        evidence_index = [evidence.index(ev) for ev in evidence if ev['image path'] in manipulation_detection_test_image_paths]
        image_to_download = [u.split(';')[0] for u in subset_evidence] #Take for each evidence the first version of the image
        for i in range(len(subset_evidence)):
            download_image(image_to_download[i],'dataset/manipulated_original_img/'+str(evidence_index[i]))
    
    #Identify originals with publication date heuristic
    dict_original_image = {}
    for img_path in manipulation_detection_test_image_paths:
        subset = [ev for ev in evidence if ev['image path'] == img_path]
        subset_index = [evidence.index(ev) for ev in evidence if ev['image path'] == img_path]
        if len(subset) > 0:
            sorted_evidence_by_date_index = pd.DataFrame(subset).sort_values(by='date').index.to_list()
            if len(sorted_evidence_by_date_index) != 0:
                idx = sorted_evidence_by_date_index[0]
                if str(idx)+'.png' in os.listdir('dataset/manipulated_original_img/'):
                    dict_original_image[img_path] = 'dataset/manipulated_original_img/'+str(subset_index[idx])+'.png'
                    

    #Save results
    with open(args.map_json_path, 'w') as file:
        json.dump(dict_original_image, file, indent=4)
