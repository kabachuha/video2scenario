import requests, json
from video_chop import chop_video
from chops_to_folder_dataset import move_the_files

# Setting up the LLM interactions
 
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'
API_KEY = "" # OpenAI or any other Auth key

with open('config.json', 'r') as cfg_file:
    config = json.loads(cfg_file.read())

with open('args.json', 'r') as cfg_file:
    args = json.loads(cfg_file.read())

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

def generate():
    ...

# Gradio interface setup if launching as an app

if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Column():
            with gr.Tabs():
                with gr.Tab(label='Level selector'):
                    # Depth slider
                    # 0 - L max
                    gr.Slider()
                    # Batch slider
                    with gr.Row():
                        # textbox with selected description
                        ...
                    ...
                with gr.Tab(label='Textgen config'):
                    with gr.Row():
                        # settings path
                        # load settings
                        # save settings
                        ...
                    with gr.Tabs():
                        with gr.Tab(label='Sampling settings'):
                            ...
                        with gr.Tab(label='Master prompts'):
                            ...
                with gr.Tab(label='Batch processing'):
                    ...
            ...
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    # L / path to video
                    ...
                with gr.Column():
                    ...
                    # splitted video folderpath
                # will chop if not exist
                ...
            with gr.Row():
                # chop video
                # clear info checkbox
                # caption keyframes checkbox
                # textgen checkbox
                # prepare dataset
                ...
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        # apply to
                        # whole
                        # level
                        #
                        ...
                    with gr.Row():
                        # generate button

                        ...
            with gr.Row():
                # list of keyframes at each selected layer
                keyframes = gr.Gallery()
        ...

    interface.launch(share=args.share, server_name=args['server_name'], server_port=args['server_port'])
