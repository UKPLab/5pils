import os
from dateutil import parser
from dateutil.tz import tzutc
import imagehash
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baseline.llm_prompting import *


def phash_image(img_path): 
    '''
    Compute perceptual hash of an image.
    '''
    img = Image.open(img_path)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    max_width = 8000
    max_height = 8000
    width, height = img.size
    if width > max_width or height > max_height:
        return None
    img_hash = imagehash.phash(img)
    return str(img_hash)


def get_duplicates(file_paths):
    '''
    Get the list of images that are duplicates of other images (excluding the first occurence of the image)
    '''
    image_hashes = []
    duplicates = []
    for file in file_paths:
        print(file)
        img_hash = phash_image(file)
        if img_hash in image_hashes:
            duplicates.append(file)
        else:
            image_hashes.append(img_hash)
    return duplicates


def get_organization(json_data):
    '''
    Get the name of the FC organization that wrote the article
    '''
    if '211' in json_data['URL']:
        org = '211org'
    elif 'factly' in json_data['URL']:
        org='factly'
    else:
        org='pesacheck'
    return org

def normalize_claim(claim_text):
    '''
    Normalize claims extracted by GPT4
    '''
    claim_text = claim_text.replace('allegedly','')
    for c in ['claims to show ', 'claimed to show ', 'claiming to show ', 'claims to depict ', 'claims to exhibit ',
              'claims that ', 'claim that ', 'claimed that ', 'the claim is that ', 'an image purportedly of ', 'claiming that ',
              'claims an ', 'purporting to show ', 'displaying ', 'claiming to be of ', 'purports to show ', 'post claimed an ',
              'alluding that ', 'post claimed ', 'post claims ', 'claimed to be of ', 'image in the post shows ', 'claims to be ',
              'on facebook claiming ', 'shared on facebook showing ', 'facebook post with an image showing ',
             'appears to show ', 'purport to show ']:
        if c in claim_text.lower():
            return claim_text.lower().split(c)[1].capitalize()
    return claim_text

def remove_vague_sources(source):    
    '''
    Remove source labels that are too vague
    '''
    vague = ['various sources','news article','news articles','multiple sources','this website', 'a website',
            'various news sources', 'various sources mentioned in the article', 'news reports']
    for v in vague:
        if v==source.lower():
            return 'not enough information'
    if 'reverse search' in source or 'reverse image search' in source:
        #Reverse image search is not a valid answer
        return 'not enough information'
    return source


def extract_named_entities(text, model, entity_type):
    '''
    Return a list of entities of a certain type contained in a string.
    Params:
        text (str) : the text string
        model (object) : the spaCy NLP model used to process the text
        entity_type (str) : the type of entity to search for. One of ["date_and_times", "locations"]
    '''
    # Process the input text using spaCy
    doc = model(text)
    # Initialize a list to store the extracted entities
    entities = []
    current_entity = []

    # Define a mapping of entity type names to spaCy labels
    entity_type_map = {
        "dates_and_times": ["DATE", "TIME"],
        "locations": ["LOC", "GPE"]
    }
    # Iterate through the tokens in the processed text
    for token in doc:
        # Check if the token is an entity of the specified type
        if token.ent_type_ in entity_type_map[entity_type]:
            if token.ent_iob_ == 'B':  # Beginning of an entity
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                current_entity.append(token.text)
            else:  # Inside or last token of an entity
                current_entity.append(token.text)
    # Add the last entity if the sentence ends with one
    if current_entity:
        entities.append(' '.join(current_entity))     
    return entities


def get_numeric_date_label(date,spacy_model):
    '''
    Convert dates to numeric labels
    '''
    date_NER = extract_named_entities(date,spacy_model, "dates_and_times")
    output=[]
    for d in date_NER:
        try:
            output.append(parser.parse(d).replace(tzinfo=tzutc()))
        except:
            pass
    if len(output)==0:
        output='not enough information'
    else:
        output = [d.isoformat() for d in output]
    return output



def get_image_path(json_data, root = 'dataset/processed_img/'):
    '''
    Obtain the local image path of a 5Pils instance
    '''
    title = json_data['URL'].split('/')[-1]
    if title + '.png' in os.listdir(root + '/'):
        image_path = root + title + '.png'
    else:
        image_path = None
    return image_path


def image_type_normalization(json_data):
    '''
    Map annotations of the type of image claim to four main categories.
    '''
    type_of_image_map = {'misattributed':'out-of-context',
                         'misidentified':'out-of-context',
                         'misrepresented':'out-of-context',
                         'misappropriated':'out-of-context',
                         'out-of-context':'out-of-context',
                         'digitally manipulated':'manipulated',
                         'digitally edited':'manipulated',
                         'morphed':'manipulated',
                         'doctored':'manipulated',
                         'altered':'manipulated',
                         'manipulated':'manipulated',
                         'fake':'fake',
                         'ai-generated':'fake',
                         'true':'true'   
                        }
    return type_of_image_map[json_data['type of image'].lower()]

def verification_strategy_normalization(json_data):
    '''
    Extract information about the verification strategies used by fact-checkers
    '''
    strategies = []
    article_text=''
    article_path = json_data['URL'].split('/')[-1] + '.txt'
    for file in os.listdir('dataset/article/'):
        if article_path in file:
            with open(os.path.join('dataset/article/', article_path),'r', encoding='utf-8') as f:
                article_text = f.read()
    for s in  ['reverse image search','keyword search','reverse search']:
        if s in article_text.lower():
            strategies.append(s)
    strategies = ['reverse image search' if s=='reverse search' else s for s in strategies ]
    if len(strategies)==0:
        strategies.append('Other/Unspecified')
    return strategies


def verification_tool_normalization(json_data):
    '''
    Extract information about the verification tools used by fact-checkers
    '''
    tools = []
    article_text=''
    article_path = json_data['URL'].split('/')[-1] + '.txt'
    for file in os.listdir('dataset/article/'):
        if article_path in file:
            with open(os.path.join('dataset/article/', article_path),'r', encoding='utf-8') as f:
                article_text = f.read()
    
    for t in  ['invid','tineye','yandex','bing','google reverse','google map','street view','google earth']:
        if t in article_text.lower():
            tools.append(t)
    if len(tools)==0:
        tools.append('Other/Unspecified')
    return tools


def is_element_in_string(string, elements):
    return any(element in string for element in elements)


def normalize_json_fields(json_data, spacy_model):
    '''
    Normalize the dataset
    '''
    #Create a copy of the original json
    normalized_json_data={}
    #Get the URL
    normalized_json_data['URL'] = json_data['URL']
    #Get image path in local files
    normalized_json_data['image path'] = get_image_path(json_data)
    #Get organization
    normalized_json_data['org'] = get_organization(json_data)
    #Get publication date
    normalized_json_data['publication date'] = json_data['publication date']
    #Get claim text 
    normalized_json_data['claim'] = normalize_claim(json_data['claim'])
    #Get provenance
    if 'yes' in str(json_data['was the photo used before?']).lower() or 'true' in str(json_data['was the photo used before?']).lower():
        normalized_json_data['was the photo used before?'] = 'yes'
    elif 'not enough' in str(json_data['was the photo used before?']).lower():
        normalized_json_data['was the photo used before?'] = 'not enough information'
    elif 'no' == str(json_data['was the photo used before?']).lower()[:2]:
        normalized_json_data['was the photo used before?']  = 'no'
    else:
        normalized_json_data['was the photo used before?'] = 'not enough information'
    #Get source
    #Remove unclear sources
    normalized_json_data['source'] = remove_vague_sources(json_data['source'])
    #Get date
    #Get numeric label
    normalized_json_data['date'] = json_data['real date']
    normalized_json_data['date numeric label'] = str(get_numeric_date_label(normalized_json_data['date'],spacy_model))
    normalized_json_data['date'] = normalized_json_data['date'] if normalized_json_data['date numeric label']!='not enough information' else 'not enough information'
    #Get location
    normalized_json_data['location'] = json_data['real location']
    #Get motivation
    normalized_json_data['motivation'] = json_data['motivation']
    #Get Type of image
    normalized_json_data['type of image'] = image_type_normalization(json_data).lower()
    #Get Verification strategy and tool
    normalized_json_data['verification strategy'] = str(verification_strategy_normalization(json_data))
    normalized_json_data['verification tool'] = str(verification_tool_normalization(json_data))
    #Normalize fields that are NEI
    for field in ['source', 'location', 'motivation']: 
        nei_str = ['not specified', 'not defined', 'not enough information', 'unknown',
                   'unspecified','not stated', 'not given', 'not mentioned', 'not provided', 'not specific',
                   'not applicable', 'unidentified', 'not applicable']
        if is_element_in_string(normalized_json_data[field].lower(),nei_str):
            normalized_json_data[field] = 'not enough information'   

    return normalized_json_data
