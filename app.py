# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.

import requests, json
from video_chop import chop_video
from chops_to_folder_dataset import move_the_files, calculate_depth, read_first_frame, read_all_frames
from video_blip2_preprocessor.preprocess import PreProcessVideos
import time, logging, coloredlogs
import os, cv2
from base64 import b64encode
from PIL import Image

logger = None

if __name__ == "__main__":
    coloredlogs.install()
    timestring = time.strftime('%Y%m%d%H%M%S')
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(f'logs/{timestring}.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    formatter_hf = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter_hf)
    logger.addHandler(fh)
    logger.addHandler(ch)

# Setting up the LLM interactions

HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate' #TODO: support multimodal interactions
API_KEY = "" # OpenAI or any other Auth key

with open('args.json', 'r') as cfg_file:
    args = json.loads(cfg_file.read())

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

def process_video():
    
    ...
    #clear video
    #chop_video
    #caption video

def run():
    print("Hey!")

# Gradio interface setup if launching as an app

if __name__ == "__main__":
    import gradio as gr

    def on_depth_change(d, L):
        return [gr.update(maximum=L**(d-1) if d > 1 else 1), gr.update(maximum=L if d > 0 else 1)]
    
    # returns depth, L, description, keyframes, base64 html
    def refresh_descr(init_path, d, scene, action):

        logger.info(f'Refreshing video tree item at {init_path}, depth {d}, part {scene}, subset {action}')

        rets = []
        assert os.path.exists(init_path) and os.path.isdir(init_path)
        # show description
        max_d, L = calculate_depth(init_path)
        rets.append(gr.update(maximum=max_d)) # update max_depth, will cause updates to other elements
        rets.append(L) # update L

        d = min(d, max_d)
        scene = min(scene, L**(d-1) if d > 1 else 1)
        action = min(action, L if d > 0 else 1)

        depth_name = init_path
        for i in range(d+1):
            depth_name = os.path.join(depth_name, f'depth_{i}')
        path = os.path.join(depth_name, f'part_{scene}')
        action_txt = os.path.join(path, f'subset_{scene}.txt')
        action_mp4 = os.path.join(path, f'subset_{action}.mp4')

        with open(action_txt, 'r', encoding='utf-8') as descr:
            rets.append(descr.read()) # descr

        if d == max_d:
            # reading all frames as keyframes
            frames = read_all_frames(action_mp4)
        else:
            next_depth_name = os.path.join(depth_name, f'depth_{d+1}')
            next_part_path = os.path.join(next_depth_name, f'part_{action+L*scene}') # `i` cause we want to sample each corresponding *subset*

            # depths > 0 are *guaranteed* to have L videos in their part_j folders
            
            # now sampling each first frame at the next level
            frames = [read_first_frame(os.path.join(next_part_path, f'subset_{k}.mp4')) for k in range(L)]
        
        frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in frames]
        frames = [Image.fromarray(img) for img in frames]
        
        rets.append(frames) # keyframes
        rets.append(action_mp4)

        return rets
    
    def write_descr(descr, init_path, depth, scene, action):
        logger.info(f'Writing video descr at {init_path}, depth {depth}, part {scene}, subset {action}')

        assert os.path.exists(init_path) and os.path.isdir(init_path)
        # show description
        max_d, L = calculate_depth(init_path)

        d = min(depth, max_d)
        scene = min(scene, L**(d-1) if d > 1 else 1)
        action = min(action, L if d > 0 else 1)

        depth_name = init_path
        for i in range(d+1):
            depth_name = os.path.join(depth_name, f'depth_{i}')
        path = os.path.join(depth_name, f'part_{scene}')
        action_txt = os.path.join(path, f'subset_{action}.txt')
        with open(action_txt, 'w', encoding='utf-8') as descr_f:
            descr_f.write(descr)
    
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row().style(equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                with gr.Tabs():
                    with gr.Tab(label='Level selector'):
                        # Depth slider
                        # 0 - L max
                        with gr.Row(variant='compact'):
                            descr_depth = gr.Slider(label="Depth", value=0, minimum=0, maximum=12, step=1, interactive=True)
                        # Batch slider
                        with gr.Row(variant='compact'):
                            descr_part = gr.Slider(label="Scene", value=0, minimum=0, maximum=12, step=1, interactive=True)
                        with gr.Row(variant='compact'):
                            descr_subset = gr.Slider(label="Action", value=0, minimum=0, maximum=12, step=1, interactive=True)
                        with gr.Row(variant='compact'):
                            # textbox with selected description
                            descr = gr.TextArea(label="Description", lines=4, interactive=True)
                        with gr.Row(variant='compact'):
                            descr_load = gr.Button('Refresh', variant='primary')
                            descr_regen_btn = gr.Button('Regenerate description')
                            descr_save_btn = gr.Button('Save description')
                    with gr.Tab(label='Textgen config'):
                        with gr.Row(variant='compact'):
                            # settings path
                            sttn_path = gr.Textbox(label="Settings path", interactive=True)
                            # load settings
                            sttn_load_btn = gr.Button('Load settings')
                            # save settings
                            sttn_save_btn = gr.Button('Save settings')
                        with gr.Tabs():
                            with gr.Tab(label='Sampling settings'):
                                gr.Markdown('Todo (see config.json)')
                                with gr.Row():
                                    textgen_url = gr.Textbox(label="Textgen URL", value="http://localhost:5000/api/v1/generate", interactive=True)
                                    textgen_key = gr.Textbox(label="API key, if private", value="", interactive=True)
                                with gr.Row():
                                    textgen_new_words = gr.Slider(label='Max new words', value=80, step=1, interactive=True, minimum=1, maximum=300)
                                    textgen_temperature = gr.Slider(label='Temperature (~creativity)', value=0.45, step=0.01, interactive=True, minimum=0, maximum=1.99)
                                with gr.Row():
                                    textgen_top_p = gr.Slider(label='Top P', value=1, step=0.01, interactive=True, minimum=0, maximum=1)
                                    textgen_typical_p = gr.Slider(label='Typical P', value=1, step=0.01, interactive=True, minimum=0, maximum=1)
                                    textgen_top_k = gr.Slider(label='Top K', value=0, step=1, interactive=True, minimum=0, maximum=100)
                                with gr.Row():
                                    textgen_repetition_penalty = gr.Slider(label='Repetition penalty', value=1.15, step=0.01, interactive=True, minimum=0, maximum=2)
                                    textgen_encoder_repetition_penalty = gr.Slider(label='Repetition penalty', value=1, step=0.01, interactive=True, minimum=0, maximum=2)
                                    textgen_length_penalty = gr.Slider(label='Length penalty', value=1, step=0.01, interactive=True, minimum=0, maximum=2)
                            with gr.Tab(label='Master prompts'):
                                with gr.Row(variant='compact'):
                                    master_scene = gr.TextArea(label="Scene", lines=5, interactive=True)
                                with gr.Row(variant='compact'):
                                    master_synopsis = gr.TextArea(label="Synopsis", lines=5, interactive=True)
                            with gr.Tab(label='Frame captioning'):
                                gr.Markdown('Frame autocaptioning (BLIP2) settings')
                                gr.Markdown('Uses bisection for more than 1 prompt/division')
                                with gr.Row(variant='compact'):
                                    autocap_frames = gr.Slider(label='Autocaptioned frames', value=2, step=1, interactive=True, minimum=1, maximum=12) # will be populater with L
                                    autocap_padding = gr.Radio(label='Padding', value='left', choices=["left", "right", "none"], interactive=True)
                                with gr.Row(variant='compact'):
                                    autocap_min_words = gr.Slider(label="Min words", minimum=1, maximum=15, value=15, step=1, interactive=True)
                                    autocap_max_words = gr.Slider(label="Max words", minimum=10, maximum=45, value=30, step=1, interactive=True)

                    with gr.Tab(label='Batch processing'):
                        gr.Markdown('Process a list of .json captioning config files:')
                        with gr.Row(variant='compact'):
                            cfgs_folder = gr.Textbox(label="Configs folder", interactive=True)
                            cfgs_start = gr.Button(value='Start', variant='primary')
                        gr.Markdown('Process a folder of videos using the current settings:')
                        with gr.Row(variant='compact'):
                            vids_folder = gr.Textbox(label="Videos folder", interactive=True)
                            vids_start = gr.Button(value='Start', variant='primary')
            with gr.Column(scale=1, variant='panel'):
                with gr.Tabs(selected=1):
                    with gr.Tab(label="Keyframes viewer"):
                        # list of keyframes at each selected layer
                        keyframes = gr.Gallery()
                        #keyframes_vid64 = gr.HTML("") # placeholder for previewable Video Base64 HTML
                        keyframes_vid64 = gr.Video(value=None, interactive=False)
                    with gr.Tab(id=1, label="Video splitter"):
                        with gr.Row(variant='compact'):
                            chop_skip_frames = gr.Slider(label='How many frames to drop from source', value=0, step=0.02, interactive=True, minimum=0, maximum=0.99)
                        # L / path to video
                            chop_L = gr.Number(label="L (each level division number)", value=12, precision=0, interactive=True)
                        with gr.Row(variant='compact'):
                            # splitted video folderpath
                            chop_whole_vid_path = gr.Textbox(label="Path to the whole video, if not splitted yet", interactive=True)
                            chop_split_path = gr.Textbox(label="Splitted video folderpath", value='split_videos/test/', interactive=True)
                            chop_trg_path = gr.Textbox(label="Target folder dataset path", interactive=True)
                            # will chop if not exist
                        with gr.Row(variant='compact'):
                            # chop video
                            do_chop = gr.Checkbox(label='(re)chop video', value=True, interactive=True)
                            # clear info checkbox
                            do_clear = gr.Checkbox(label='clear info', interactive=True)
                        with gr.Row(variant='compact'):
                            # caption keyframes checkbox
                            do_caption = gr.Checkbox(label='caption keyframes', value=True, interactive=True)
                            # textgen checkbox
                            do_textgen = gr.Checkbox(label='textgen scenes', value=True, interactive=True)
                        with gr.Row(variant='compact'):
                            # export checkbox
                            do_export = gr.Checkbox(label='export to dataset', interactive=True)
                            do_delete = gr.Checkbox(label='delete after export', interactive=True)
                        with gr.Row(variant='compact'):
                            with gr.Column(variant='compact'):
                                with gr.Row(variant='compact'):
                                    # apply to
                                    # whole
                                    # this level
                                    #
                                    do_apply_to = gr.Radio(label="Apply to:", value="Whole video", choices=["Whole video", "This level"], interactive=True)
                            with gr.Column(variant='compact'):
                                with gr.Row(variant='compact'):
                                    # generate button
                                    do_btn = gr.Button('Process', variant="primary")
                            do_infobox = gr.Markdown('', visible=False)

                    with gr.Tab(label="Video export settings"):
                        exp_overwrite_dims = gr.Checkbox(label="Override dims", value=True, interactive=True)
                        exp_w = gr.Slider(label="Width", value=768, minimum=64, maximum=1920, step=64, interactive=True)
                        exp_h = gr.Slider(label="Height", value=432, minimum=64, maximum=1920, step=64, interactive=True)
                        exp_overwrite_fps = gr.Checkbox(label="Override fps", value=False, interactive=True)
                        exp_fps = gr.Slider(label="FPS", value=12, minimum=1, maximum=144, step=1, interactive=True)
        
        # interactions
        descr_depth.change(on_depth_change, inputs=[descr_depth, chop_L], outputs=[descr_part, descr_subset])
        descr_load.click(refresh_descr, outputs=[descr_depth, chop_L, descr, keyframes, keyframes_vid64], inputs=[chop_split_path, descr_depth, descr_part, descr_subset])
        descr_save_btn.click(write_descr, inputs=[descr, chop_split_path, descr_depth, descr_part, descr_subset], outputs=[])
        #depth, L, description, video, keyframes, gallery, base64 html

    interface.launch(share=args["share"], server_name=args['server_name'], server_port=args['server_port'])
