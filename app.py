import requests, json
from video_chop import chop_video
from chops_to_folder_dataset import move_the_files

# Setting up the LLM interactions
 
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'
API_KEY = "" # OpenAI or any other Auth key

with open('textgen_config.json', 'r') as cfg_file:
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
    print("Hey!")

# Gradio interface setup if launching as an app

if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row().style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                with gr.Tabs():
                    with gr.Tab(label='Level selector'):
                        # Depth slider
                        # 0 - L max
                        with gr.Row(variant='compact'):
                            gr.Slider(label="Depth", minimum=0, maximum=12, step=1)
                        # Batch slider
                        with gr.Row(variant='compact'):
                            gr.Slider(label="Batch", minimum=0, maximum=144, step=1)
                        with gr.Row(variant='compact'):
                            # textbox with selected description
                            gr.TextArea(label="Description", lines=4, interactive=True)
                        with gr.Row(variant='compact'):
                            gr.Button('Regenerate description')
                    with gr.Tab(label='Textgen config'):
                        with gr.Row(variant='compact'):
                            # settings path
                            gr.Textbox(label="Settings path", interactive=True)
                            # load settings
                            gr.Button('Load settings')
                            # save settings
                            gr.Button('Save settings')
                        with gr.Tabs():
                            with gr.Tab(label='Sampling settings'):
                                gr.Markdown('Todo (see config.json)')
                            with gr.Tab(label='Master prompts'):
                                with gr.Row(variant='compact'):
                                    gr.TextArea(label="Scene", lines=5, interactive=True)
                                with gr.Row(variant='compact'):
                                    gr.TextArea(label="Synopsis", lines=5, interactive=True)
                                with gr.Row(variant='compact'):
                                    gr.Markdown('Frame autocaptioning (BLIP2) settings')
                                with gr.Row(variant='compact'):
                                    gr.Slider(label="Min prompt words", minimum=1, maximum=15, value=15, step=1)
                                    gr.Slider(label="Max prompt words", minimum=10, maximum=45, value=30, step=1)

                    with gr.Tab(label='Batch processing'):
                        gr.Markdown('Todo')
            with gr.Column(scale=1, variant='panel'):
                with gr.Tabs():
                    with gr.Tab(label="Keyframes viewer"):
                        # list of keyframes at each selected layer
                        keyframes = gr.Gallery()
                    with gr.Tab(label="Video splitter"):
                        with gr.Row(variant='compact'):
                            with gr.Column(variant='compact'):
                                # L / path to video
                                with gr.Row(variant='compact'):
                                    gr.Number(label="L", value=12, precision=0)
                                with gr.Row(variant='compact'):
                                    gr.Textbox(label="Path to whole video", interactive=True)
                            with gr.Column(variant='compact'):
                                # splitted video folderpath
                                gr.Textbox(label="Splitted video folderpath", interactive=True)
                            # will chop if not exist
                        with gr.Row(variant='compact'):
                            # chop video
                            gr.Checkbox(label='chop video')
                            # clear info checkbox
                            gr.Checkbox(label='clear info')
                            # caption keyframes checkbox
                            gr.Checkbox(label='caption keyframes')
                            # textgen checkbox
                            gr.Checkbox(label='textgen scenes')
                        with gr.Row(variant='compact'):
                            with gr.Column(variant='compact'):
                                with gr.Row(variant='compact'):
                                    # apply to
                                    # whole
                                    # this level
                                    #
                                    gr.Radio(label="Apply to:", value="Whole video", choices=["Whole video", "This level"])
                            with gr.Column(variant='compact'):
                                with gr.Row(variant='compact'):
                                    # generate button
                                    gen_btn = gr.Button('Start', variant="primary")
                    with gr.Tab(label="Export"):
                        with gr.Row(variant='compact'):
                            gr.Textbox(label="Target folder dataset path", interactive=True)
                            data_btn = gr.Button('Compose dataset', variant="primary")

    interface.launch(share=args["share"], server_name=args['server_name'], server_port=args['server_port'])
