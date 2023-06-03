import gradio as gr
import requests, json
from video_chop import chop_video
from chops_to_folder_dataset import move_the_files

# Setting up the LLM interactions
 
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'
API_KEY = "" # OpenAI or any other Auth key

with open('config.json', 'r') as cfg_file:
    config = json.loads(cfg_file.read())

HOST = config.pop('host')
API_KEY = config.pop('api_key')

def textgen(prompt):
    with open('config.json', 'r') as cfg_file:
        config = json.loads(cfg_file.read())

    assert config is not None
    config.pop('host')

    request = config
    request['prompt'] = prompt

    print(request)

    result = ''

    try:
        response = requests.post(URI, json=request, headers={'Content-Type':'application/json', 'Authorization': 'Bearer {}'.format(API_KEY)})
        if response.status_code == 200:
            result = response.json()['results'][0]['text']
            print(result)
        else:
            raise Exception(f'Request returned status {response.status_code}')
    except Exception as e:
            print(e)
            raise e
    return result


