import Levenshtein as lev
from dateutil.tz import tzutc
from dateutil import parser
import requests
from trafilatura import bare_extraction
from bs4 import BeautifulSoup as bs
import pandas as pd
from PIL import Image
from io import BytesIO
import requests as rq
import os
import time
import numpy as np
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *


def scrape_image(url,article):
    '''
    Scrape an image given its url and store it locally as a png file.
    '''
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        req = rq.get(url, stream=True, timeout=(10,10), headers=headers)
    except:
        return None
    if req.status_code == 200 and 'image' in req.headers.get('Content-Type', ''):
        image_content = req.content
        image = Image.open(BytesIO(image_content))
        image.verify()
        with Image.open(BytesIO(image_content)) as img:
            file_path = 'dataset/img/'
            img_file_name = article[:-4]
            img.save(file_path + img_file_name + '.png')


def load_urls(file_path):
    '''
    Load all fact-checking articles URLs as a list
    '''
    with open(file_path, 'r') as file:
        # Read the lines and create a list
        url_list = [line.strip() for line in file]
        #Remove duplicates
        url_list = list(set(url_list))
        file.close()
    return url_list


def is_english_article(url):
    '''
    Verify if the  article is saved in English. 
    Some articles of Factly are written in Kannada and Telugu.
    Some articles of Pesacheck are written in French.
    '''
    if 'telugu' in url or 'kannada' in url:
        return False
    if 'faux' in url or "intox" in url or 'ces-photo' in url or 'cette-photo' in url or 'cette-image' in url or 'ces-images' in url:
        return False
    return True


def pesacheck_parser(url):
    '''
    Scrape an URL using request and parse it with BeautifulSoup to collect the FC article data and the image url.
    Custom script for Pesacheck articles.
    '''
    try:
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'} 
        req = rq.get(url,headers=headers).text
    except : 
        return '', []
    soup = bs(req, 'html.parser')
    title = soup.title.text + '\n'
    pub_date_tag = soup.find('meta', {'property': 'article:published_time'})

    if pub_date_tag:
        publication_date = "Publication Date:" + pub_date_tag.get('content', 'No publication date found')
    else:
        publication_date = "Publication date not found in the HTML."
    filtered_paragraphs = soup.find_all('p')
    try:
        text = title + '\n' + publication_date + '\n' + '\n'.join([p.get_text() for p in filtered_paragraphs]).split('This post is part of an ongoing series of PesaCheck')[0].split('--')[1]
    except:
        text = title + '\n' + publication_date + '\n' + '\n'.join([p.get_text() for p in filtered_paragraphs])
    image_urls = [img['srcset']  for img in  soup.find_all('source', type='image/webp')]
    try:
        image_urls = [image_urls[1].split(',')[0].split()[0]]
    except:
        image_urls = ''
    text += '\nImage URLs :\n' + '\n'.join(image_urls)
    return text, image_urls


def two11org_parser(url):
    '''
    Scrape an URL using request and parse it with BeautifulSoup to collect the FC article data and the image url.
    Custom script for 211Check articles.
    '''
    try:
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'} 
        req = rq.get(url,headers=headers).text
    except : 
        return '', []
    soup = bs(req, 'html.parser')
    title = soup.title.text + '\n'
    pub_date_tag = soup.find('meta', {'property': 'article:published_time'})

    if pub_date_tag:
        publication_date = "Publication Date:" + pub_date_tag.get('content', 'No publication date found')
    else:
        publication_date = "Publication date not found in the HTML."
    filtered_paragraphs = soup.find_all('p')
    text = title + '\n' + publication_date + '\n' + '\n'.join([p.get_text() for p in filtered_paragraphs]).split('Name *')[0]
    image_urls = [[img['src'] for img in soup.find_all('img')][2]]
    text += '\nImage URLs :\n' + '\n'.join(image_urls)

    return text, image_urls


def factly_parser(url):
    '''
    Scrape an URL using request and parse it with BeautifulSoup to collect the FC article data and the image url.
    Custom script for Factly articles.
    '''
    try:
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'} 
        req = rq.get(url,headers=headers).text
    except : 
        return '', [] 
    soup = bs(req, 'html.parser')
    title = soup.title.text + '\n'
    pub_date_tag = soup.find('meta', {'property': 'article:published_time'})

    if pub_date_tag:
        publication_date = "Publication Date:" + pub_date_tag.get('content', 'No publication date found')
    else:
        publication_date = "Publication date not found in the HTML."
    filtered_paragraphs = soup.find_all('p')
    text = title + '\n' + publication_date + '\n' + '\n'.join([p.get_text() for p in filtered_paragraphs]).split('FACTLY is one of the well known Data Journalism/Public Information portals in India.')[0]
    image_urls = [img['src']  for img in  soup.find_all('img')]
    image_urls = [i for i in image_urls if 'logo' not in i.lower() and 'thumbnail' not in i.lower()][5:6]
    text += '\nImage URLs :\n' + '\n'.join(image_urls)
    return text, image_urls


def collect_articles(urls,
                     parser, 
                     scrape_images=True, 
                     image_urls=None,
                     sleep=10):
    '''
    Collect the fact-checking articles and images based on their URLs.
    '''
    img_urls_unique = set()
    if 'article' not in os.listdir('dataset/'):
        os.mkdir('dataset/article/')
    for u in range(len(urls)):
        files = [f.split('.txt')[0] for f in os.listdir('dataset/article/')]
        is_new_article=True
        if  urls[u].split('/')[-1].split('?')[0]  in files:
            is_new_article = False
            print('Already scraped : ' + urls[u].split('/')[-1].split('?')[0])
        if is_new_article:
            path = 'dataset/article/'+ urls[u].split('/')[-1].split('?')[0] + '.txt'
            text, scraped_image_urls = parser(urls[u]) #Use a platform specific parser
            scraped_image_urls = [img for img in scraped_image_urls if img not in img_urls_unique]
            for img in scraped_image_urls:
                    img_urls_unique.add(img)    
            #Save text
            with open(path,'w',encoding='utf-8') as f:
                text = 'URL: ' + urls[u] + '\n' + text
                f.write(text)
            if scrape_images:
                #Scrape the image and save the content
                if 'img' not in os.listdir('dataset/'):
                    os.mkdir('dataset/img/')
                if image_urls!= None:
                    #A reference image url is already provided as part of the dataset
                    scrape_image(image_urls[u], path.split('/')[-1])
                    time.sleep(3)
                else:
                    #If no existing image urls are provided, default to the scraped ones
                    for im_url in scraped_image_urls:
                        scrape_image(im_url, path.split('/')[-1])
                        time.sleep(3)
            time.sleep(sleep)


def is_obfuscated_or_encoded(url):
    '''
    Check that the evidence url is not obfuscated or encoded.
    '''
    unquoted_url = url
    try:
        return '%' in unquoted_url or '//' in unquoted_url.split('/')[2]
    except:
        return True


def is_likely_html(url):
    '''
    Check that the evidence url is html
    '''
    # List of common file extensions
    file_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.doc', '.docx', '.ppt', '.pptx', '.xls', 
                       '.xlsx', '.txt', '.zip', '.rar', '.exe', '.svg', '.mp4', '.avi', '.mp3']

    # Extract the extension from the URL
    extension = '.' + url.rsplit('.', 1)[-1].lower()

    # Check if the URL ends with a common file extension
    if extension in file_extensions:
        return False
    else:
        return True
    

def is_fc_organization(url):
    '''
    Check that the evidence url does not come from a FC organization
    Note: the provided list does not include every single existing FC organization. Some FC articles might still pass through this filter.
    '''
    fc_domains = ['https://www.fastcheck.cl','https://pesacheck.org','https://africacheck.org','https://www.snopes.com',
            'https://newsmobile.in', 'https://211check.org', 'factcrescendo.com/', 'https://leadstories.com', 'https://www.sochfactcheck.com', 
            'https://newschecker.in','https://www.altnews.in', 'https://dubawa.org', 'https://factcheck.afp.com', 'factly.in', 
            'https://misbar.com/factcheck/', 'larepublica.pe/verificador/', 'fatabyyano.net/', 'https://www.vishvasnews.com/', "newsmeter.in" , 
            "boomlive", "politifact","youturn.in", "lemonde.fr/les-decodeurs","factuel.afp.com","thequint.com", "logicalindian.com/fact-check/", 
            "timesofindia.com/times-fact-check", "indiatoday.in/fact-check/", "smhoaxslayer.com", "facthunt.in", "aajtak.in/fact-check/",
            "bhaskar.com/no-fake-news", "theprint.in/hoaxposed/", 'firstdraftnews.org']
    for d in fc_domains :
        if d in url:
            return True
    return False


def is_banned(url):
    '''
    Check if the evidence url is in the list of banned urls
    '''
    banned = [
        #Those websites are flagged as potential unsafe or block the webscraping process
        "legalaiddc-prod.qed42.net", "windows8theme.net", "darkroom.baltimoresun.com", "dn.meethk.com", "hotcore.info", "pre-inscription.supmti.ac.ma",
        "hideaways.in", "www.alhurra.com/search?st=articleEx", "anonup.com", "hiliventures", "middleagerealestateagent", "nonchalantslip.fr",
        "corseprestige.com", ".aa.com.tr",  "landing.rassan.ir", "aiohotzgirl.com", "hotzxgirl.com",
        #The content of those social media websites is not scrapable.
        "facebook.com", "twitter.c", "youtube.co", "linkedin.co", "tiktok.c", "quora.c", "gettyimages.", "reddit." ]
    for b in banned:
        if b in url:
            return True
    return False


def get_filtered_retrieval_results(path):
    '''
    Filter the results of reverse image search.
    Args:
        path (str): path to the file that contains the raw RIS results from Google Reverse Image Search
    '''
    ris_results = load_json(path)
    retrieval_results = []
    # Iterate over the URLs and apply the filters
    for i in range(len(ris_results)):
        for u in range(len(ris_results[i]['urls'])):
            #Loop through all evidence urls, and see if they meet the requirements
            evidence_url = ris_results[i]['urls'][u]
            ris_data = {
                'image path': ris_results[i]['image path'], 
                'raw url': evidence_url,
                'image urls': ris_results[i]['image urls'][evidence_url], 
                'is_fc': is_fc_organization('/'.join(evidence_url.split('/')[:3])),
                'is_https': evidence_url.startswith('https')
            }
            # Apply additional conditions to each dictionary
            ris_data['is_banned'] = is_banned(ris_data['raw url'])
            ris_data['is_obfuscated'] = is_obfuscated_or_encoded(ris_data['raw url'])  
            ris_data['is_html'] = is_likely_html(ris_data['raw url'])
            # Selection condition
            ris_data['selection'] = ris_data['is_html'] and ris_data['is_https'] and not ris_data['is_obfuscated'] and not ris_data['is_banned']
            # Append the dictionary to the list if it meets all the criteria
            retrieval_results.append(ris_data)

    # Filter the data based on the selection criteria
    selected_retrieval_results = [d for d in retrieval_results if d['selection']]
    return selected_retrieval_results


def compute_url_distance(url1,url2,threshold):
    distance = lev.distance(url1,url2)
    if distance < threshold:
        return True
    else:
        return False


def find_image_caption(soup, image_url,threshold=25):
    '''
    Retrieve the caption corresponding to an image url by searching the html in BeautifulSoup format.
    '''
    img_tag = None
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-original')
        if src and compute_url_distance(src, image_url, threshold):
            img_tag = img
            break
    if not img_tag:
        return "Image not found"
    figure = img_tag.find_parent('figure')
    if figure:
        figcaption = figure.find('figcaption')
        if figcaption:
            return figcaption.get_text().strip()
    for sibling in img_tag.find_next_siblings(['div', 'p','small']):
        if sibling.get_text().strip():
            return sibling.get_text().strip()
    title = img_tag.get('title')
    if title:
        return title.strip()
    # Strategy 4: Use the alt attribute of the image
    alt_text = img_tag.get('alt')
    if alt_text:
        return alt_text.strip()

    return "Caption not found"


def extract_info_trafilatura(page_url,image_url):
    try:
        headers= {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'} 
        response = requests.get(page_url, headers=headers, timeout=(10,10))
        if response.status_code == 200:
            #Extract content with Trafilatura
            result = bare_extraction(response.text,
                                   include_images=True,
                                   include_tables=False)
            #Remove unnecessary contente
            keys_to_keep = ['title','author','url',
                            'hostname','description','sitename',
                            'date','text','language','image','pagetype']
            result = {key: result[key] for key in keys_to_keep if key in result}
            result['image url'] = image_url
            # Finding the image caption
            image_caption = []
            soup = bs(response.text, 'html.parser')
            for img in image_url:
                image_caption.append(find_image_caption(soup, img))
            image_caption.append(find_image_caption(soup,result['image']))
            result['image caption'] = image_caption
            result['url'] = page_url
            return result
        else:
            return "Failed to retrieve webpage"
    except Exception as e:
        return f"Error occurred: {e}"


def time_difference(date1, date2):
    '''
    Compute whether date1 preceeds date2
    '''
    # Parse the dates
    dt1 = parser.parse(date1)
    dt2 = parser.parse(date2)
    # Make both dates offset-aware, assuming UTC if no timezone is provided
    if dt1.tzinfo is None:
        dt1 = dt1.replace(tzinfo=tzutc())
    if dt2.tzinfo is None:
        dt2 = dt2.replace(tzinfo=tzutc())
    return dt1 < dt2


def merge_data(evidence, evidence_metadata,dataset):
    '''
    Merge all evidence by dropping duplicates and applying 2 filters:
    1) The evidence is not the original FC article itself
    2) The evidence has been published before the FC article
    '''
    evidence_df = pd.DataFrame(evidence)
    evidence_metadata_df = pd.DataFrame(evidence_metadata)
    dataset_df = pd.DataFrame(dataset)
    merged_data = pd.merge(evidence_df, evidence_metadata_df.drop_duplicates(subset='raw url')[['image path','raw url']].rename(columns={'raw url':'url'}), on='url',how='inner')
    merged_data = pd.merge(merged_data.rename(columns={'url':'evidence url'}), 
                           dataset_df[['org','image path','publication date']].rename(columns={'publication date': 'date_filter'}), 
                           on='image path',how='inner')
    merged_data  = merged_data.dropna(subset='evidence url')
    #Verify that the evidence is not the FC article itself.
    fc_mask = merged_data.apply(lambda row : False if row['org'] in row['evidence url'] or row['org'] in ''.join(row['image url']) else True, axis=1)
    merged_data = merged_data[fc_mask]
    #Remove evidence that have been published after the FC article or have no publication date
    merged_data = merged_data[~merged_data['date'].isnull()]
    time_mask = merged_data.apply(lambda row : True if time_difference(row['date'],row['date_filter']) else False,axis=1)
    merged_data = merged_data[time_mask]   
    merged_data = merged_data[['image path','org','evidence url','title','author','hostname',
                           'description','sitename','date','image','image url','image caption']]
    merged_data = merged_data.drop_duplicates(subset=['evidence url','image path'])
    return merged_data


def download_image(url, file_path, max_size_mb=10):
    '''
    Download evidence images. Only used for images predicted as manipulated to replace them by their predicted original version.
    '''
    try:
        # Send a GET request to the URL
        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = rq.get(url, stream=True, timeout=(10,10),headers=headers)
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to download. Status code: {response.status_code}")
            return None
        # Check the content type to be an image
        if 'image' not in response.headers.get('Content-Type', ''):
            print("URL does not point to an image.")
            return None
        # Check the size of the image
        if int(response.headers.get('Content-Length', 0)) > max_size_mb * 1024 * 1024:
            print(f"Image is larger than {max_size_mb} MB.")
            return None
        # Read the image content
        image_data = response.content
        if not image_data:
            print("No image data received.")
            return None
        image = Image.open(BytesIO(image_data))
        image.verify()
        image = Image.open(BytesIO(image_data))
        # Save the image to a file
        image.save(file_path + '.png')
        print("Image downloaded and saved successfully.")
    except rq.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def keep_longest_non_greyscale_area(image, color_threshold=30):
    '''
    Automatically remove social media sidebars for screenshots coming from social media platforms.
    '''
    def find_longest_sequence(arr, color_threshold):
        # Calculate the range (max-min) within each row for each color channel
        color_ranges = np.ptp(arr, axis=1)
        # Determine rows that have enough color variation to not be considered greyscale
        is_colorful = np.any(color_ranges > color_threshold, axis=1)
        # Identify transitions between greyscale and non-greyscale rows
        transitions = np.diff(is_colorful.astype(int))
        start_indices = np.where(transitions == 1)[0] + 1  # Start of colorful sequence
        end_indices = np.where(transitions == -1)[0]  # End of colorful sequence
        # Handle cases where the sequence starts from the first row or ends at the last row
        if len(start_indices) == 0 or (len(end_indices) > 0 and start_indices[0] > end_indices[0]):
            start_indices = np.insert(start_indices, 0, 0)
        if len(end_indices) == 0 or (len(start_indices) > 0 and end_indices[-1] < start_indices[-1]):
            end_indices = np.append(end_indices, arr.shape[0] - 1)
        # Find the longest sequence of colorful rows
        max_length = 0
        max_seq_start = max_seq_end = 0
        for start, end in zip(start_indices, end_indices):
            if end - start > max_length:
                max_length = end - start
                max_seq_start, max_seq_end = start, end
        # Keep only the longest colorful sequence
        return arr[max_seq_start:max_seq_end]

    # Convert to numpy array for analysis
    image_array = np.array(image)

    # Keep the longest sequence of non-greyscale rows
    longest_row_sequence = find_longest_sequence(image_array, color_threshold)

    # Transpose the array to treat columns as rows and repeat the process
    transposed_array = np.transpose(longest_row_sequence, (1, 0, 2))
    longest_col_sequence = find_longest_sequence(transposed_array, color_threshold)

    # Transpose back to original orientation
    final_array = np.transpose(longest_col_sequence, (1, 0, 2))

    # Convert the array back to an image
    longest_sequence_image = Image.fromarray(final_array)
    return longest_sequence_image


def apply_instructions(image, instructions):
    '''
    Applies processing instructions to the given image.
    Instructions include standard processing, cropping, or downloading a new image and then cropping.
    '''
    if instructions.startswith("Standard processing"):
        return keep_longest_non_greyscale_area(image)  
    elif instructions.startswith("Cropped"):
        crop_coords = eval(instructions.split(": ")[1])  
        return image.crop(crop_coords)
    elif instructions.startswith("Replaced with URL"):
        parts = instructions.split("; ")
        url = parts[0].split(": ")[1]
        new_image = download_image(url) 
        if len(parts) > 1 and parts[1].startswith("Standard processing"):
            return keep_longest_non_greyscale_area(image)
        if len(parts) > 1 and parts[1].startswith("Cropped"):
            crop_coords = eval(parts[1].split(": ")[1])
            return new_image.crop(crop_coords)
        return new_image
    else:
        return image  


def process_images_from_instructions(instructions_file, source_folder, target_folder):
    '''
    Loads processing instructions from a file and applies them to each corresponding image.
    '''
    with open(instructions_file, 'r') as file:
        for line in file:
            image_name, instructions = line.strip().split(": ", 1)
            image_path = os.path.join(source_folder, image_name)    
            # Check if the image exists before processing
            if not os.path.exists(image_path):
                continue       
            image = Image.open(image_path)
            processed_image = apply_instructions(image, instructions)    
            # Save the processed image to the target folder
            target_image_path = os.path.join(target_folder, image_name)
            processed_image.save(target_image_path)
            print(f"Processed and saved: {image_name}")
