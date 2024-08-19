from openai import AzureOpenAI
import os
import time
import argparse
from tqdm import tqdm
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from baseline.llm_prompting import *

def label_corpus(corpus, 
                 client, 
                 system_prompt_path,
                 json_path,
                 sleep=20):
    '''
    Loop over the FC articles and use GPT4 to extract the relevant data as a structured json file.
    Params:
        corpus (list): list of strings containing the articles content
        client (object): the Azure OpenAI client
        system_prompt_path (str): The prompt providing instructions to GPT4 for data labeling
        json_path (str): the path to the directory where gpt4 annotations will be stored
        sleep (int): waiting time between two API calls
    '''
    system_prompt = open(system_prompt_path).read()
    for t in tqdm(range(len(corpus))): 
        content = system_prompt + '\n\nArticle: \n'+ corpus[t]
        output, _ = gpt4_prompting(content,client)
        if type(output)==str:
            save_result(output,json_path)
            time.sleep(sleep)

            

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Annotate the FC articles with GPT4.')
    parser.add_argument('--image_dir_path', type=str, default='dataset/img/',
                        help='The path to the directory where images are stored.')   
    parser.add_argument('--article_dir_path', type=str, default='dataset/article/',
                        help='The path to the directory where articles are stored.') 
    parser.add_argument('--json_file_path', type=str, default='dataset/gpt4_annotations/annotations.json',
                        help='The path to the directory where gpt4 annotations will be stored.')  
    #The system prompt is based on the excellent tutorial and template provided by Matthias Bastian 
    # on https://the-decoder.com/how-openai-prompts-its-own-custom-chatgpts/   
    parser.add_argument('--system_prompt', type=str, default='dataset_collection/system_prompt.txt',
                        help='The prompt providing instructions to GPT4 for data labeling.')  
    parser.add_argument('--openai_api_key', type=str, default = " ", #Insert your key here
                        help='The environment variable name for your secret key to access Azure openAI services.')
    parser.add_argument('--api_version', type=str, default = "2023-10-01-preview",
                        help='The version of the Azure OpenAI services to use.')
    parser.add_argument('--endpoint', type=str, default = " ", #Insert your endpoint here
                        help='The environment variable name for the endpoint to access Azure openAI services.')

    args = parser.parse_args()

    if 'gpt4_annotations' not in os.listdir('dataset'):
        os.mkdir('dataset/gpt4_annotations/')

    #Get the OpenAI Azure client
    client = AzureOpenAI(
                api_key=os.getenv(args.openai_api_key),  
                api_version=args.api_version,
                azure_endpoint = os.getenv(args.endpoint)
                )
    #Get corpus
    corpus = get_corpus(args.article_dir_path, args.json_file_path,args.image_dir_path)
    label_corpus(corpus, client, args.system_prompt,args.json_file_path)  