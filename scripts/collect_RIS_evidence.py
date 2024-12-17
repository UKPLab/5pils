from google.cloud import vision
import os 
from tqdm import tqdm
import time
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *
import argparse



def detect_web(path,how_many_queries=30):
    """
    Detects web annotations given an image.
    """
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image, max_results=how_many_queries)
    annotations = response.web_detection

    page_urls = []
    matching_image_urls = {}
    visual_entities = {}

    if annotations.pages_with_matching_images:
        print(
            "\n{} Pages with matching images found:".format(
                len(annotations.pages_with_matching_images)
            )
        )
        
        for page in annotations.pages_with_matching_images:
            page_urls.append(page.url)
            if page.full_matching_images:
                #List of image URLs for that webpage (the image can appear more than once)
                matching_image_urls[page.url] = [image.url for image in page.full_matching_images]
            else:
                matching_image_urls[page.url] = []
            if page.partial_matching_images: 
                matching_image_urls[page.url] += [image.url for image in page.partial_matching_images] 
    else:
        print('No matching images found for ' + path)
    if annotations.web_entities:
        for entity in annotations.web_entities:
            #Collect web entities as entity-score dictionary pairs
            visual_entities[entity.description] = entity.score

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    
    return page_urls, matching_image_urls, visual_entities



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Collect evidence using Google Reverse Image Search.')
    parser.add_argument('--collect_google', type=int, default=0, 
                        help='Whether to collect evidence URLs with the google API. If 0, it is assumed that a file containing URLs already exists.')
    parser.add_argument('--evidence_urls', type=str, default='dataset/retrieval_results/evidence_urls.json',
                        help='Path to the list of evidence URLs to scrape. Needs to be a valid file if collect_google is set to 0.')
    parser.add_argument('--google_vision_api_key', type=str,  default= " ", #Provide your own key here as default value
                        help='Your key to access the Google Vision services, including the web detection API. Only needed if collect_google is set to 1.')  
    parser.add_argument('--image_path', type=str, default='dataset/processed_img/',
                        help='The folder where the images are stored.') 
    parser.add_argument('--raw_ris_urls_path', type=str, default='dataset/retrieval_results/ris_results.json',
                        help='The json file to store the raw RIS results.') 
    parser.add_argument('--scrape_with_trafilatura', type=int, default=1, 
                        help='Whether to scrape the evidence URLs with trafilatura. If 0, it is assumed that a file containing the scraped webpages already exists.') 
    parser.add_argument('--trafilatura_path', type=str, default='dataset/retrieval_results/trafilatura_data.json',
                        help='The json file to store the scraped trafilatura  content as a json file.')
    parser.add_argument('--apply_filtering', type=int, default=0,
                        help='If 1, remove evidence published after the source FC article. Not needed if using the default evidence set')
    parser.add_argument('--json_path', type=str, default='dataset/retrieval_results/evidence.json',
                        help='The json file to store the text evidence as a json file.')
    parser.add_argument('--max_results', type=int, default=50,
                        help='The maximum number of web-pages to collect with the web detection API.') 
    parser.add_argument('--sleep', type=int, default=3,
                        help='The waiting time between two web detection API calls') 
    

    args = parser.parse_args()
    key = os.getenv(args.google_vision_api_key)

    #Create directories if they do not exist yet
    if not 'retrieval_results'  in os.listdir('dataset/'):
        os.mkdir('dataset/retrieval_results/')

    #Google RIS
    if args.collect_google:
        #Change the output file
        raw_ris_results = []
        for path in tqdm(os.listdir(args.image_path)):
            urls, image_urls, vis_entities  = detect_web(args.image_path +path, args.max_results)
            raw_ris_results.append({'image path':args.image_path + path, 
                                    'urls': urls, 
                                    'image urls': image_urls,  
                                    'visual entities': vis_entities
                                    }
            )
            time.sleep(args.sleep)
        with open(args.raw_ris_urls_path, 'w') as file:
            #Save raw results
            json.dump(raw_ris_results, file, indent=4)
        #Apply filtering to the URLs to remove content produced by FC organizations and content that is not scrapable
        selected_data = get_filtered_retrieval_results(args.raw_ris_urls_path)

    else:
        #Load evidence that have already been collected 
        #Further ensure that there is a corresponding image already downloaded
        selected_data = [d for d in load_json(args.evidence_urls) if d['image path'].split('/')[-1] in os.listdir('dataset/processed_img/')]
        
    urls = [d['raw url'] for d in selected_data]
    images = [d['image urls'] for d in selected_data]

    if args.scrape_with_trafilatura:
        #Collect results with Trafilatura
        output = []
        for u in tqdm(range(len(urls))):
            output.append(extract_info_trafilatura(urls[u],images[u]))
            #Only store in json file every 50 evidence
            if u%50==0:
                save_result(output,args.trafilatura_path) 
                output = []
    
    #Save all results in a Pandas Dataframe
    evidence_trafilatura = load_json(args.trafilatura_path)
    dataset = load_json('dataset/train.json') + load_json('dataset/val.json')  + load_json('dataset/test.json')
    evidence = merge_data(evidence_trafilatura, selected_data, dataset, apply_filtering=args.apply_filtering).fillna('').to_dict(orient='records')
    # Save the list of dictionaries as a JSON file
    with open(args.json_path, 'w') as file:
        json.dump(evidence, file, indent=4)
