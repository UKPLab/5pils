import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset_collection.scrape_utils import *
from utils import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Download articles and images based on provided URLs.')
    parser.add_argument('--file_path', type=str, default='dataset/url/article_urls.txt',
                        help='File path of the URLs to scrape.')
    parser.add_argument('--scrape_image', type=int, default=1,
                        help='If 1, downloads the FC images in addition to the article text. If 0, assumes the images have already been scraped.') 
    parser.add_argument('--image_processing_script', type=str, default='dataset/image_processing_instructions.txt',
                        help='Script with automated instructions to crop and clean the images.') 
    parser.add_argument('--sleep', type=int, default=5,
                        help='Waiting time in seconds between collection of articles.')

    args = parser.parse_args()
    #Collect the article and image for the 1676 instances of the 5Pillars dataset
    urls = open(args.file_path,'r').read().split('\n')
    #Scrape the article content and the images
    factly_urls = [u for u in urls if 'factly.in' in u]
    collect_articles(factly_urls,factly_parser,args.scrape_image,args.sleep)
    pesacheck_urls = [u for u in urls if 'pesacheck.org' in u]
    collect_articles(pesacheck_urls,pesacheck_parser,args.scrape_image,args.sleep)
    two11org_urls = [u for u in urls if '211check.org' in u]
    collect_articles(two11org_urls,two11org_parser,args.scrape_image,args.sleep)
    #Image processing
    if not 'processed_img' in os.listdir('dataset/'):
        os.mkdir('dataset/processed_img/')
    process_images_from_instructions(args.image_processing_script, 'dataset/img/', 'dataset/processed_img/')
