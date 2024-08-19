import argparse
from openai import AzureOpenAI
import os
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from baseline.answer_generation import * 
from baseline.generation_utils import * 
from baseline.llm_prompting import *


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate 5 pillars answers with LLMs.')
    parser.add_argument('--openai_api_key', type=str, default =' ', #Insert your key here
                        help='The environment variable name for your secret key to access Azure openAI services.')
    parser.add_argument('--api_version', type=str, default ='2023-10-01-preview',
                        help='The version of the Azure OpenAI services to use.')
    parser.add_argument('--endpoint', type=str, default =' ', #Insert your endpoint here
                        help='The environment variable name for the endpoint to access Azure openAI services.')
    parser.add_argument('--map_manipulated_original', type=str, default='dataset/map_manipulated_original.json',
                        help='Path to the file that maps manipulated images to their identified original version.')
    parser.add_argument('--results_file', type=str, default='output/results.json',
                        help='Path to store the predicted answers.')
    parser.add_argument('--task', type=str, default='source',
                        help='The task to perform. One of [source, date, location, motivation]')
    parser.add_argument('--modality', type=str, default='vision',
                        help='Which input modality to use. One of [vision, evidence, multimodal]')
    parser.add_argument('--n_shots', type=int, default=0,
                        help='How many demonstrations to include.')
    parser.add_argument('--model', type=str, default='llava',
                        help='Which LLM to use for generating answers.')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='The maximum number of tokens to generate as output.')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='The temperature of the model. Lower values make the output more deterministic.')
    parser.add_argument('--sleep', type=int, default=5,
                        help='The waiting time between two answer generation.')

    args = parser.parse_args()
    if args.model=='gpt4':
        client = AzureOpenAI(
        api_key=os.getenv(args.openai_api_key),  
        api_version=args.api_version,
        azure_endpoint = os.getenv(args.endpoint)
        )
    else:
        client = None

    if 'output' not in os.listdir():
        os.mkdir('output/')
    map_manipulated = load_json(args.map_manipulated_original)
    try:
        results_json = load_json(args.results_file)
    except:
        # file does not exist yet
        results_json = []
    #Prepare data
    train = load_json('dataset/train.json')
    #Load test images
    test = load_json('dataset/test.json')
    task_test = [t for t in test if t[args.task]!='not enough information']
    image_paths = [t['image path'] for t in task_test]
    ground_truth = [t[args.task] for t in task_test]

    #Load embeddings and evidence
    clip_evidence_embeddings = np.load('dataset/embeddings/evidence_embeddings.npy')
    image_embeddings = np.load('dataset/embeddings/image_embeddings.npy')
    image_embeddings_map = load_json('dataset/embeddings/image_embeddings_map.json')
    evidence = load_json('dataset/retrieval_results/evidence.json')
    #Select evidence and demonstrations
    evidence_idx = []
    if args.modality in ['evidence','multimodal']:
        for i in range(len(image_paths)):
            evidence_idx.append(get_topk_evidence(image_paths[i], evidence, image_embeddings, clip_evidence_embeddings, image_embeddings_map))
    #Select demonstrations
    #Keep train images that have evidence
    images_with_evidence = [ev['image path'] for ev in evidence]
    demonstration_candidates = [t for t in train if t['image path'] in images_with_evidence]
    # print(demonstration_candidates)
    demonstrations = []
    for i in range(len(image_paths)):
        if args.n_shots > 0:

            demonstrations_idx = get_topk_demonstrations(image_paths[i],args.task,demonstration_candidates,
                                                            image_embeddings,image_embeddings_map,args.n_shots)
            instance_demonstrations = []
            for idx in demonstrations_idx:
                #Get the top k evidence for each demonstration
                demo_image = demonstration_candidates[idx]['image path']
                demo_answer = demonstration_candidates[idx][args.task]
                demo_evidence_idx = get_topk_evidence(demo_image,evidence,image_embeddings, clip_evidence_embeddings, image_embeddings_map)
                demo_evidence = [evidence[idx] for idx in demo_evidence_idx]
                instance_demonstrations.append((demo_image, demo_answer, demo_evidence))
            demonstrations.append(instance_demonstrations)
        else:
            demonstrations.append([])
    
    #Run the main loop
    run_model(image_paths, 
              args.task, 
              ground_truth, 
              results_json,
              map_manipulated, 
              args.modality, 
              args.model, 
              evidence, 
              evidence_idx, 
              demonstrations, 
              client,
              args.max_tokens, 
              args.temperature, 
              args.sleep)
