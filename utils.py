import os
import json
import base64


def load_json(file_path):
    '''
    Load json file
    '''
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def concatenate_entry(d):
    '''
    For all keys in a dictionary, if a value is a list, concatenate it.
    '''
    for key, value in d.items():
        if isinstance(value, list):  
            d[key] = ';'.join(map(str, value))  # Convert list to a string separated by ';'
    return d


def append_to_json(file_path, data):
    '''
    Append a dict or a list of dicts to a JSON file.
    '''
    try:
        if not os.path.exists(file_path):
            # Create an empty JSON file with an empty list if it does not exist yet
            with open(file_path, 'w') as file:
                json.dump([], file)
        #Open the existing file
        with open(file_path, 'r+') as file:
            file_data = json.load(file)
            if type(data)==list:
                for d in data:
                    if type(d)==dict:
                        file_data.append(concatenate_entry(d))
            else:
                file_data.append(concatenate_entry(data))
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")


def save_result(output,json_file_path):
    '''
    Save output results to a JSON file.
    '''
    try:    
        if type(output)==str:
            output = json.loads(output)
            append_to_json(json_file_path, output)
        else:
            append_to_json(json_file_path, output)
    except json.JSONDecodeError:
        #The output was not well formatted
        pass


def entry_exists(json_file_path, url):
    '''
    Check if an entry for the given URL already exists in the JSON file.
    '''
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            return any(entry.get("URL").split('/')[-1] == url.split('/')[-1].split('.')[0] for entry in data)
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not a valid JSON file.")
        return False
    except FileNotFoundError:
        return False


def is_folder_empty(folder_path):
    '''
    Check if the given folder is empty.
    '''
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        return not os.listdir(folder_path)


def get_corpus(directory, json_file_path,image_directory):
    '''
    Process each text file in the given directory.
    '''
    text_files = []
    corpus = []
    # Identify the text files
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            text_files.append(os.path.join(directory, file))
    # Process each text file
    for txt_file in text_files:
        txt_file_name = os.path.basename(txt_file)
        image_folder_name = txt_file_name[:-4]  # Remove '.txt'
        image_folder_path = os.path.join(image_directory, image_folder_name)

        if is_folder_empty(image_folder_path):
            continue
            
        if entry_exists(json_file_path, txt_file):
            continue
        with open(txt_file, 'r',encoding='utf-8') as file:
            text = file.read()
            text = text.split('Image URLs')[0]
            corpus.append(text)
    return corpus


def encode_image(image_path):
  '''
  Encode images in base64. Format required by GPT4-Vision.
  '''
  with open(image_path, "rb") as image_file:
    return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
