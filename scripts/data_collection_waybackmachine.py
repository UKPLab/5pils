import requests as rq
import time
import json  
from tqdm import tqdm
import os
import argparse
import re
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *


def collect_data_wayback(website_url,
                 output_dir,
                 start_date,
                 end_date,
                 resume_key='',
                 max_count=1000,
                 chunk_size=100,
                 sleep=3):
    '''
    Collect all urls matching a specific domain on the Wayback machine.
    All archived urls between a specified start date and end date are returned in alphabetical order.
    It is important to not overload the API by keeping the chunk_size parameter reasonably low, 
    and waiting for a few seconds between each API call.
    Params:
        website_url (str): the url domain. All urls matching that domain will be searched on the Wayback machine.
        output_dir (str): the path to the file were the retrieved urls are stored.
        start_date (int): results archived from that date will be returned. Format : YYYYMMDD
        end_date (int): results archived up to that date will be returned. Format : YYYYMMDD
        resume_key (str): if not all urls have been returned in the previous iteration, the resume key allows to start from the last url retrieved.
        max_count (int): the maximum number of results to be returned.
        chunk_size (int): the number of results to return per batch. 
        sleep (int): waiting time between API calls. 
    '''
    
    if chunk_size > max_count:
        return ValueError('Chunk size needs to be smaller than max count.')
    unique_articles_set = set()
    url_list = []
    #Define the search URL to query the Wayback Machine API
    url = 'http://web.archive.org/cdx/search/cdx?url=https://www.' + website_url + '&collapse=digest&filter=!statuscode:404&showResumeKey=true&matchType=prefix&from=%s&to=%s&limit=%s&output=json'%(start_date,end_date,chunk_size)
    if resume_key!='':
        #Resume from a certain point, if a resume key is provided
        url += '&resumeKey='+resume_key
    its = max_count//chunk_size 
    for _ in tqdm(range(its)):
        #Main loop : cycle until we collect sufficient articles
        result_urls = rq.get(url).text
        parse_url = json.loads(result_urls) #parses the JSON from urls.
        resume_key = parse_url[-1][0]
        ## Extracts timestamp and original columns from urls and compiles a url list.
        for i in range(1,len(parse_url)-2):
            orig_url = parse_url[i][2]
            if len(orig_url.lower().split(website_url)) > 1:
                article_name = orig_url.lower().split(website_url)[1].split('/')[0].split('?')[0]
            else:
                article_name = website_url   
            if  article_name not in unique_articles_set:
                tstamp = parse_url[i][1]
                waylink ='https://web.archive.org/web/' + tstamp+'/'+orig_url.split(website_url)[0] + website_url + article_name
                url_list.append(waylink)
                unique_articles_set.add(article_name)
        url =  'http://web.archive.org/cdx/search/cdx?url=' + website_url + '&collapse=digest&showResumeKey=true&resumeKey='+ resume_key +'&matchType=prefix&filter=!statuscode:404&from=%s&to=%s&limit=%s&output=json'%(start_date,end_date,chunk_size)       
        time.sleep(sleep)     

    # save the url list as a txt file if they match the keywords

    print('Collected %s of the initial number of requested urls'%(round(len(url_list)/max_count,2)))
    with open(output_dir, 'a') as file:
        for item in url_list:
                file.write(str(item) + '\n')
        file.write(resume_key)
    print('Last article : ' + url_list[-1].split('/')[-2] + '/' + url_list[-1].split('/')[-1])
    print('Enter resume key if you want to continue.')
    return resume_key




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Download articles and images from the Wayback machine.')
    parser.add_argument('--url_domain', type=str, default='factly.in/', 
                        help='The domain to query on the Wayback Machine API. factly.in/, 211check.org/, or pesacheck.org/')
    parser.add_argument('--org', type=str, default='factly',
                        help='FC organization from which articles are collected')
    parser.add_argument('--file_path', type=str, default='dataset/url/factly.txt',
                        help='path to the file that stores the URLs')
    parser.add_argument('--parser', default=factly_parser,
                        help='A parser function. Each FC organization has a dedicated one. They are found in scrape_utils.py.')   
    parser.add_argument('--scrape_image', type=int, default=1,
                        help='If True, downloads the FC images in addition to the article text.') 
    parser.add_argument('--process_image', type=int, default=1,
                        help='If True, clean and crop images to remove social media sidebars.') 
    parser.add_argument('--image_processing_script', type=str, default='dataset/image_processing_instructions.txt',
                        help='Script with automated instructions to crop and clean the images. Needs to be a valid file with instructions if process_image is True') 
    parser.add_argument('--start_date', type=int, default=20190101,
                        help='Start date for the collection of URLS.')
    parser.add_argument('--end_date', type=int, default=20231231,
                        help='End date for the collection of URLS.')
    parser.add_argument('--max_count', type=int, default=20000,
                        help='Maximum number of URLs to collect.')
    parser.add_argument('--chunk_size', type=int, default=4000,
                        help='Size of each chunk to query the Wayback Machine API.')
    parser.add_argument('--sleep', type=int, default=5,
                        help='Waiting time between two calls of the Wayback machine API.')

    args = parser.parse_args()


    collect_data_wayback(args.url_domain,  
                        args.file_path,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        max_count=args.max_count,
                        chunk_size=args.chunk_size,
                        sleep=args.sleep)
    #Remove non-English and non-images FC article URLs
    search_words= ["photo", "image", "picture"]
    pattern = r'(?:' + '|'.join(search_words) + r')'
    urls = [u for u in load_urls(args.file_path) if is_english_article(u) and re.search(pattern,u)]
    #Remove the wayback machine part of the URL
    original_urls = ['/'.join(u.split('/')[5:]) for u in urls] 
    #Scrape the article content and the images
    collect_articles(original_urls,args.parser,args.scrape_image,args.sleep)
    #Process the images
    if args.process_image:
        if not 'processed_img' in os.listdir('dataset/'):
            os.mkdir('dataset/processed_img/')
        process_images_from_instructions(args.image_processing_script, 'dataset/img/', 'dataset/processed_img/')