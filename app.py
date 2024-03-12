# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.

from typing import Union
import requests, json

from torch.autograd import variable
from video_chop import chop_video
from chops_to_folder_dataset import move_the_files, calculate_depth, read_first_frame, read_all_frames
import time, logging, coloredlogs, random, math
import os, cv2
from base64 import b64encode
import torch, gc
from PIL import Image
import shutil
from tqdm import tqdm

def do_seed(seed):
    if seed == -1:
        random.seed(time.time())
    else:
        random.seed(seed)

# Gradio interface setup if launching as an app

if __name__ == "__main__":
    with open('args.json', 'r') as cfg_file:
        args = json.loads(cfg_file.read())
    
    def textgen(prompt, **params): # 1.03.24 change: switched to OpenAI Oobabooga API
        # NOTE reopening every time to allow dynamic changes to params
        with open('textgen_config.json', 'r') as cfg_file:
            config = json.loads(cfg_file.read())

        assert config is not None
        URL = params.pop('textgen_url') # e.g. "http://127.0.0.1:5000/v1/chat/completions"
        API_KEY = params.pop('textgen_key')

        for k, v in params.items():
            config[k] = v
        
        # the config.json can now have HISTORY

        if not 'messages' in config:
            config['messages'] = []
        
        config['messages'].append({"role": "user", "content": prompt})
        
        logger.info('Sending textgen request to server')
        logger.debug(config)

        result = ''

        try:
            response = requests.post(URL, json=config, headers={'Content-Type':'application/json', 'Authorization': 'Bearer {}'.format(API_KEY)}, verify=False)
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                print(result)
            else:
                raise Exception(f'Request returned status {response.status_code}')
        except Exception as e:
                print(e)
                raise e
        return result
    
    def video_llava_gen(video_path, llava_url, llava_prompt):
        
        logger.info('Sending captioning request to LLaVA-server')
        logger.info(f'Of video at {video_path}')

        file = {'file': open(video_path, 'rb')}
        #data = {'prompt': llava_prompt}

        result = ''

        try:
            response = requests.post(llava_url, files=file)#, headers={'Content-Type':'multipart/form-data'}, verify=True)
            if response.status_code == 200:
                result = response.json()['message']
                #print(result)
            else:
                raise Exception(f'Request returned status {response.status_code}')
        except Exception as e:
                print(e)
                raise e
        return result

    logger = None

    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    coloredlogs.install()
    timestring = time.strftime('%Y%m%d%H%M%S')
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{logs_dir}/{timestring}.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    formatter_hf = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter_hf)
    logger.addHandler(fh)
    logger.addHandler(ch)
    import gradio as gr

    with open('master_prompt_scene.txt', 'r', encoding='utf-8') as cfg_file:
        master_scene_default = cfg_file.read()
    
    with open('master_prompt_synopsis.txt', 'r', encoding='utf-8') as cfg_file:
        master_synopsis_default = cfg_file.read()
    
    with open('master_prompt_llava.txt', 'r', encoding='utf-8') as cfg_file:
        master_llava_default = cfg_file.read()
    
    def process_video(do_chop, do_clear, do_caption, do_textgen, do_export, do_delete, chop_L, input_video_path, split_video_path, dataset_path, textgen_url, textgen_key, leaves_dropout_factor, seed, master_scene, master_synopsis, exp_overwrite_dims, exp_w, exp_h, exp_overwrite_fps, exp_fps, video_llava_url, master_llava_prompt):
        L = chop_L
        logger.info(f'Processing video at {input_video_path}')
        do_seed(seed)

        #chop video
        if do_chop:
            if os.path.exists(split_video_path):
                shutil.rmtree(split_video_path)
            chop_video(input_video_path, split_video_path, L)

        max_d, L = calculate_depth(split_video_path)
        #max_d = max_d - 1

        # caption video
        if do_caption:
            logger.info(f'Captioning frames')

            try:
                # arrive at ground frames to caption them with blip
                depth_name = split_video_path
                for i in range(max_d):
                    depth_name = os.path.join(depth_name, f'depth_{i}')
                for j in range(L**(max_d-1) if max_d > 1 else 1):
                    part_path = os.path.join(depth_name, f'part_{j}')
                    amount_of_completions = 0

                    # sample the text info for the next subset
                    subset_len = L if max_d > 0 else 1
                    for i in range(subset_len):
                        if random.uniform(0, 1) <= leaves_dropout_factor**max_d \
                            or amount_of_completions == 0 and i >= math.floor(subset_len / 4) and i <= math.ceil(3*subset_len / 4): # or always caption in the middle

                            txt_path = os.path.join(part_path, f'subset_{i}.txt')
                            mp4_path = os.path.join(part_path, f'subset_{i}.mp4')

                            #image = read_first_frame(mp4_path)
                            #descr = "" # caption_image(image, beam_amount, min_length, max_length)

                            descr = video_llava_gen(mp4_path, video_llava_url, master_llava_prompt)

                            with open(txt_path, 'w' if do_clear else 'a', encoding='utf-8') as descr_f:
                                descr_f.write(descr)
                            amount_of_completions += 1

            except Exception as e:
                logger.error(e)

        if do_textgen:
            t_counter=0
            for d in range(0, max_d-1):
                for j in range(L**(d-1) if d > 1 else 1):
                    for i in range(L if d > 0 else 1):
                        t_counter+=1
            tq = tqdm(total=t_counter)

            logger.info(f'Generating descriptions')

            # going reverse here
            assert max_d-1>0
            for d in range(max_d-1,-1,-1):
                depth_name = split_video_path
                for i in range(d if d > 0 else 1):
                    depth_name = os.path.join(depth_name, f'depth_{i}')
                for j in range(L**(d-1) if d > 2 else 1):
                    part_path = os.path.join(depth_name, f'part_{j}')
                    amount_of_completions = 0
                    # sample the text info for the next subset
                    subset_len = L if d > 1 else 1
                    for i in range(subset_len):
                        txt_path = os.path.join(part_path, f'subset_{i}.txt')
                        mp4_path = os.path.join(part_path, f'subset_{i}.mp4')
                        tq.set_description(f'Depth {d}, part {j}, subset{i}')

                        with open(txt_path, 'r', encoding='utf-8') as descr_f:
                            peek_d = descr_f.read()
                        if len(peek_d) > 0 and not do_clear:
                            continue
                        
                        if random.uniform(0, 1) <= leaves_dropout_factor**d \
                            or amount_of_completions == 0 and i >= math.floor(subset_len / 4) and i <= math.ceil(3*subset_len / 4): # or always caption in the middle
                            next_depth_name = os.path.join(depth_name, f'depth_{d if d > 0 else 1}')
                            next_part_path = os.path.join(next_depth_name, f'part_{i+L*j}') # `i` cause we want to sample each corresponding *subset*

                            # depths > 0 are *guaranteed* to have L videos in their part_j folders
                            
                            # now sampling each description at the next level
                            scenes = ''
                            for k in range(L):
                                subset_path = os.path.join(next_part_path, f'subset_{k}.txt')
                                if os.path.exists(subset_path):
                                    with open(subset_path, 'r', encoding='utf-8') as subdescr:
                                        scenes += subdescr.read() + '\n'
                                else:
                                    if leaves_dropout_factor == 1:
                                        raise Exception('Previous level subset was not generated while the dropout factor is 1!')
                            
                            if d == 0:
                                prompt = master_synopsis.replace('%descriptions%', scenes)
                            else:
                                prompt = master_scene.replace('%descriptions%', scenes)
                            
                            textgen_json = {"textgen_url":textgen_url, "textgen_key":textgen_key}#{"max_new_tokens":max_new_tokens, "temperature":temperature, "top_p":top_p, "typical_p":typical_p, "top_k":top_k}

                            descr = textgen(prompt, **textgen_json)

                            with open(txt_path, 'w', encoding='utf-8') as descr_f:
                                descr_f.write(descr)

                        tq.update(1)
            
            tq.close()
        
        if do_export:
            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)
    
            move_the_files(split_video_path, dataset_path, L, max_d-1, exp_overwrite_dims, exp_w, exp_h, exp_overwrite_fps, exp_fps)

            if do_delete:
                shutil.rmtree(split_video_path)

    def on_depth_change(d, L, s, a):
        new_s = L**(d-1)-1 if d > 1 else 0
        new_a = L-1 if d > 0 else 1
        return [gr.update(maximum=new_s, value=min(s, new_s)), gr.update(maximum=new_a, value=min(a, new_a))]
    
    # returns depth, L, description, keyframes, base64 html
    def refresh_descr(init_path, d, scene, action):

        logger.info(f'Refreshing video tree item at {init_path}, depth {d}, part {scene}, subset {action}')

        rets = []
        assert os.path.exists(init_path) and os.path.isdir(init_path)
        # show description
        max_d, L = calculate_depth(init_path)
        max_d = max_d - 1
        rets.append(gr.update(maximum=max_d)) # update max_depth, will cause updates to other elements
        rets.append(L) # update L

        d = min(d, max_d)
        scene = min(scene, L**(d-1) if d > 1 else 1)
        action = min(action, L if d > 0 else 1)

        depth_name = init_path
        for i in range(d+1):
            depth_name = os.path.join(depth_name, f'depth_{i}')
        path = os.path.join(depth_name, f'part_{scene}')
        action_txt = os.path.join(path, f'subset_{action}.txt')
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
        max_d = max_d - 1

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
    
    def descr_regen(init_path, depth, scene, action, master_scene, master_synopsis, master_llava_prompt, video_llava_url, textgen_url, textgen_key, leaves_dropout_factor, seed):

        #textgen_json = locals()
        #for i in 'init_path,depth,scene,action,beam_amount,min_length,max_length,master_scene,master_synopsis'.split(','):
        #    textgen_json.pop(i)
        logger.warning(f'ReWriting video descr at {init_path}, depth {depth}, part {scene}, subset {action}')

        assert os.path.exists(init_path) and os.path.isdir(init_path)
        # show description
        max_d, L = calculate_depth(init_path)
        max_d = max_d - 1

        d = min(depth, max_d)
        scene = min(scene, L**(d-1) if d > 1 else 1)
        action = min(action, L if d > 0 else 1)

        depth_name = init_path
        for i in range(d+1):
            depth_name = os.path.join(depth_name, f'depth_{i}')
        path = os.path.join(depth_name, f'part_{scene}')
        action_mp4 = os.path.join(path, f'subset_{action}.mp4')

        # use BLIP2 for ground frame captioning, use LLMs for upper level
        # NOTE 1.03.24 - BLIP2 deprecated, using videollava server instead
        if d == max_d:
            descr = video_llava_gen(action_mp4, video_llava_url, master_llava_prompt)
            # image = read_first_frame(action_mp4)
            # descr = descr_regen_image(image, beam_amount, min_length, max_length)
        else:
            next_depth_name = os.path.join(depth_name, f'depth_{d+1}')
            next_part_path = os.path.join(next_depth_name, f'part_{action+L*scene}') # `i` cause we want to sample each corresponding *subset*

            # depths > 0 are *guaranteed* to have L videos in their part_j folders
            
            # now sampling each description at the next level
            scenes = ''
            for k in range(L):
                subset_path = os.path.join(next_part_path, f'subset_{k}.txt')
                if os.path.exists(subset_path):
                    with open(subset_path, 'r', encoding='utf-8') as subdescr:
                        scenes += subdescr.read() + '\n'
                else:
                    if leaves_dropout_factor == 1:
                        raise Exception('some subparts are absent while no dropout is set!')
            
            prompt = master_synopsis.replace('%descriptions%', scenes)
            if d == 0:
                prompt = master_synopsis.replace('%descriptions%', scenes)
            else:
                prompt = master_scene.replace('%descriptions%', scenes)
            
            textgen_json = {"textgen_url":textgen_url, "textgen_key":textgen_key}
            descr = textgen(prompt, **textgen_json)
        
        return descr

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
                                    textgen_url = gr.Textbox(label="Textgen URL", value="http://127.0.0.1:5001/v1/chat/completions", interactive=True)
                                    textgen_key = gr.Textbox(label="API key, if private", value="", interactive=True)
                                #with gr.Row():
                                #    textgen_new_words = gr.Slider(label='Max new words', value=80, step=1, interactive=True, minimum=1, maximum=300)
                                #    textgen_temperature = gr.Slider(label='Temperature (~creativity)', value=0.45, step=0.01, interactive=True, minimum=0, maximum=1.99)
                                #with gr.Row():
                                #    textgen_top_p = gr.Slider(label='Top P', value=1, step=0.01, interactive=True, minimum=0, maximum=1)
                                #    textgen_typical_p = gr.Slider(label='Typical P', value=1, step=0.01, interactive=True, minimum=0, maximum=1)
                                #    textgen_top_k = gr.Slider(label='Top K', value=0, step=1, interactive=True, minimum=0, maximum=100)
                                #with gr.Row():
                                #    textgen_repetition_penalty = gr.Slider(label='Repetition penalty', value=1.15, step=0.01, interactive=True, minimum=0, maximum=2)
                                #    textgen_encoder_repetition_penalty = gr.Slider(label='Repetition penalty', value=1, step=0.01, interactive=True, minimum=0, maximum=2)
                                #    textgen_length_penalty = gr.Slider(label='Length penalty', value=1, step=0.01, interactive=True, minimum=0, maximum=2)
                            with gr.Tab(label='Master prompts'):
                                with gr.Row(variant='compact'):
                                    master_scene = gr.TextArea(label="Scene", lines=5, interactive=True, value=master_scene_default)
                                with gr.Row(variant='compact'):
                                    master_synopsis = gr.TextArea(label="Synopsis", lines=5, interactive=True, value=master_synopsis_default)
                                with gr.Row(variant='compact'):
                                   master_llava_prompt = gr.TextArea(label="VideoLLaVA prompt", lines=5, interactive=True, value=master_llava_default)
                            with gr.Tab(label='Frame captioning'):
                                gr.Markdown('Frame autocaptioning (VideoLLAVA) settings')
                                with gr.Row(variant='compact'):
                                    video_llava_url = gr.Textbox(label="VideoLLaVA server ip", value="http://127.0.0.1:7861/describe")
                                
                                # with gr.Row(variant='compact'):
                                #     autocap_frames = gr.Slider(label='Autocaptioned frames', value=2, step=1, interactive=True, minimum=1, maximum=12) # will be populater with L
                                #     autocap_beam_amount = gr.Slider(label='Beam amount', value=7, step=1, interactive=True, minimum=1, maximum=30)
                                # with gr.Row(variant='compact'):
                                #     autocap_min_words = gr.Slider(label="Min words", minimum=1, maximum=15, value=15, step=1, interactive=True)
                                #     autocap_max_words = gr.Slider(label="Max words", minimum=10, maximum=45, value=30, step=1, interactive=True)

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
                            leaves_dropout_factor = gr.Slider(label='Leaves to remain at each level^D', value=0, step=0.8, interactive=True, minimum=0, maximum=1)
                        # L / path to video
                            chop_L = gr.Number(label="L (each level division number)", value=12, precision=0, interactive=True)
                            seed = gr.Number(label="Seed for dropout", value=-1, precision=0, interactive=True)
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
                                    do_apply_to = gr.Radio(label="Apply to:", value="Whole video", choices=["Whole video", "This level"], interactive=False)
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
        descr_depth.change(on_depth_change, inputs=[descr_depth, chop_L, descr_part, descr_subset], outputs=[descr_part, descr_subset])
        descr_load.click(refresh_descr, outputs=[descr_depth, chop_L, descr, keyframes, keyframes_vid64], inputs=[chop_split_path, descr_depth, descr_part, descr_subset])
        descr_regen_btn.click(descr_regen, inputs=[chop_split_path, descr_depth, descr_part, descr_subset, master_scene, master_synopsis, master_llava_prompt, video_llava_url, textgen_url, textgen_key, leaves_dropout_factor, seed], outputs=[descr])
        descr_save_btn.click(write_descr, inputs=[descr, chop_split_path, descr_depth, descr_part, descr_subset], outputs=[])

        # process
        do_btn.click(process_video, inputs=[do_chop, do_clear, do_caption, do_textgen, do_export, do_delete, chop_L, chop_whole_vid_path, chop_split_path, chop_trg_path, textgen_url, textgen_key, leaves_dropout_factor, seed, master_scene, master_synopsis, exp_overwrite_dims, exp_w, exp_h, exp_overwrite_fps, exp_fps, video_llava_url, master_llava_prompt])

    if 'api' in args:
        api_port = args["api_port"]
        import uvicorn, fire
        from typing import Union, Any, Dict, List
        from fastapi import FastAPI, Query, Request, UploadFile
        from fastapi.encoders import jsonable_encoder
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
        from fastapi import HTTPException
        from json import JSONDecodeError
        from concurrent.futures import ThreadPoolExecutor
        app = FastAPI()

        api_executor = ThreadPoolExecutor(max_workers=1) # concurrency locked to 1
        submitted_jobs: Dict[str, Any] = {}

        @app.post("/process")
        def describe_uploaded_video(dataset_id:str, file: UploadFile, chop_L: int, do_chop: bool = True, do_clear: bool = True, do_caption: bool = True, do_textgen: bool = True, do_export: bool = True, do_delete: bool = True, chop_split_path: Union[str, None] = None, chop_trg_path: Union[str, None] = None, textgen_url: Union[str, None] = None, textgen_key: Union[str, None] = None, leaves_dropout_factor: float = 1.0, seed: int = -1, master_scene: Union[str, None] = None, master_synopsis: Union[str, None] = None, exp_overwrite_dims: bool = False, exp_w: int = 512, exp_h: int = 288, exp_overwrite_fps: bool = False, exp_fps: int = 12, video_llava_url: Union[str, None] = None, master_llava_prompt: Union[str, None] = None):

            video_file = file
            logger.debug(f'Incoming video {video_file.filename}!')
            filename = "temp."+str(video_file.filename).split('.')[-1]
            try:
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(video_file.file, f)
            except Exception:
                return JSONResponse(
                    status_code=422,
                    content=jsonable_encoder({"message": "There was an error uploading the file"}),
                )
            finally:
                video_file.file.close()

            if textgen_url is None:
                textgen_url = "http://127.0.0.1:5001/v1/chat/completions"
            if textgen_key is None:
                textgen_key = ""
            
            if master_scene is None:
                master_scene = master_scene_default
            if master_synopsis is None:
                master_synopsis = master_synopsis_default
            if master_llava_prompt is None:
                master_llava_prompt = master_llava_default
            if video_llava_url is None:
                master_llava_prompt = "http://127.0.0.1:7861/describe"

            # chop_whole_vid_path Where to copy
            # chop_trg_path
            # chop_split_path

            chop_whole_vid_path = filename

            chop_split_path = os.path.join(os.getcwd(), 'datasets', 'splits', dataset_id)
            chop_trg_path = os.path.join(os.getcwd(), 'datasets', 'exported', dataset_id)
            
            os.makedirs(chop_split_path, exist_ok=True)
            os.makedirs(chop_trg_path, exist_ok=True)

            def submit_wrapper(dataset_id, *args, **kwargs):
                logger.info(f'Starting job {dataset_id}!')
                process_video(*args, **kwargs)
                logger.info(f'Job {dataset_id} completed!')

            future = api_executor.submit(lambda: submit_wrapper(dataset_id, do_chop, do_clear, do_caption, do_textgen, do_export, do_delete, chop_L, chop_whole_vid_path, chop_split_path, chop_trg_path, textgen_url, textgen_key, leaves_dropout_factor, seed, master_scene, master_synopsis, exp_overwrite_dims, exp_w, exp_h, exp_overwrite_fps, exp_fps, video_llava_url, master_llava_prompt))
        
        uvicorn.run(app, port=api_port)

    interface.launch(share=args["share"], server_name=args['server_name'], server_port=args['server_port'])
