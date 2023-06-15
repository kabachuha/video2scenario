# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.

import requests, json
from video_chop import chop_video
from chops_to_folder_dataset import move_the_files, calculate_depth, read_first_frame, read_all_frames
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time, logging, coloredlogs
import os, cv2
from base64 import b64encode
import torch, gc
from PIL import Image
import shutil
from tqdm import tqdm

logger = None

if __name__ == "__main__":

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

# Setting up the LLM interactions

with open('args.json', 'r') as cfg_file:
    args = json.loads(cfg_file.read())

def textgen(prompt, params):
    with open('textgen_config.json', 'r') as cfg_file:
        config = json.loads(cfg_file.read())

    assert config is not None
    URL = params.pop('textgen_url')
    API_KEY = params.pop('textgen_key')

    for k, v in params.items():
        config[k] = v

    config['prompt'] = prompt

    logger.info('Sending textgen request to server')
    logger.debug(config)

    result = ''

    try:
        response = requests.post(URL, json=config, headers={'Content-Type':'application/json', 'Authorization': 'Bearer {}'.format(API_KEY)})
        if response.status_code == 200:
            result = response.json()['results'][0]['text']
            print(result)
        else:
            raise Exception(f'Request returned status {response.status_code}')
    except Exception as e:
            print(e)
            raise e
    return result

# Gradio interface setup if launching as an app

if __name__ == "__main__":
    import gradio as gr
    
    blip_model = None
    processor = None

    with open('master_prompt_scene.txt', 'r', encoding='utf-8') as cfg_file:
        master_scene_default = cfg_file.read()
    
    with open('master_prompt_synopsis.txt', 'r', encoding='utf-8') as cfg_file:
        master_synopsis_default = cfg_file.read()

    def load_blip():
        global processor, blip_model
        if processor is not None and blip_model is not None:
            return
        print("Loading BLIP2")

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        blip_model = model
    
    def unload_blip():
        global blip_model, processor
        blip_model = None
        processor = None
        torch.cuda.empty_cache()
        gc.collect()
    
    # utility function
    def caption_image(image, beam_amount, min_length, max_length):
        global processor, blip_model
        inputs = processor(images=image, return_tensors="pt").to(blip_model.device, torch.float16)
        generated_ids = blip_model.generate(
                **inputs, 
                num_beams=beam_amount, 
                min_length=min_length, 
                max_length=max_length
            )
        generated_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True)[0].strip()
        
        return generated_text
    
    def process_video(do_chop, do_clear, do_caption, do_textgen, do_export, do_delete, input_video_path, split_video_path, dataset_path, beam_amount, min_length, max_length, textgen_url, textgen_key, max_new_tokens, temperature, top_p, typical_p, top_k, repetition_penalty, encoder_repetition_penalty, length_penalty, master_scene, master_synopsis, exp_overwrite_dims, exp_w, exp_h, exp_overwrite_fps, exp_fps):
        
        input_video_path = input_video_path.value
        split_video_path = split_video_path.value
        
        logger.info(f'Processing video at {input_video_path}')

        max_d, L = calculate_depth(input_video_path)
        max_d = max_d - 1

        #chop video
        if do_chop:
            if os.path.exists(split_video_path):
                shutil.rmtree(split_video_path)
            chop_video(input_video_path, split_video_path, L)

        # caption video
        if do_caption:
            logger.info(f'Captioning frames')
            load_blip()

            try:
                # arrive at ground frames to caption them with blip
                depth_name = split_video_path
                for i in range(max_d):
                    depth_name = os.path.join(depth_name, f'depth_{i}')
                for j in range(L**(max_d-1) if max_d > 1 else 1):
                    part_path = os.path.join(depth_name, f'part_{j}')
                    # sample the text info for the next subset
                    for i in range(L if max_d > 0 else 1):
                        txt_path = os.path.join(part_path, f'subset_{i}.txt')
                        mp4_path = os.path.join(part_path, f'subset_{i}.mp4')

                        image = read_first_frame(mp4_path)
                        descr = caption_image(image, beam_amount, min_length, max_length)

                        with open(txt_path, 'w' if do_clear else 'a', encoding='utf-8') as descr_f:
                            descr_f.write(descr)

            except Exception as e:
                logger.error(e)
            finally:
                unload_blip()

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
            for d in range(range(max_d-1,-1,-1)):
                depth_name = split_video_path
                for i in range(d):
                    depth_name = os.path.join(depth_name, f'depth_{i}')
                for j in range(L**(d-1) if d > 1 else 1):
                    part_path = os.path.join(depth_name, f'part_{j}')
                        # sample the text info for the next subset
                    for i in range(L if d > 0 else 1):
                        txt_path = os.path.join(part_path, f'subset_{i}.txt')
                        mp4_path = os.path.join(part_path, f'subset_{i}.mp4')
                        tq.set_description(f'Depth {d}, part {j}, subset{i}')

                        with open(txt_path, 'r', encoding='utf-8') as descr_f:
                            peek_d = descr_f.read()
                        if len(peek_d) > 0 and not do_clear:
                            continue

                        next_depth_name = os.path.join(depth_name, f'depth_{d+1}')
                        next_part_path = os.path.join(next_depth_name, f'part_{i+L*j}') # `i` cause we want to sample each corresponding *subset*

                        # depths > 0 are *guaranteed* to have L videos in their part_j folders
                        
                        # now sampling each description at the next level
                        scenes = ''
                        for k in range(L):
                            with open(os.path.join(next_part_path, f'subset_{k}.txt'), 'r', encoding='utf-8') as subdescr:
                                scenes += subdescr.read() + '\n'
                        
                        if d == 0:
                            prompt = master_synopsis.replace('%descriptions%', scenes)
                        else:
                            prompt = master_scene.replace('%descriptions%', scenes)
                        
                        textgen_json = {"textgen_url":textgen_url, "textgen_key":textgen_key, "max_new_tokens":max_new_tokens, "temperature":temperature, "top_p":top_p, "typical_p":typical_p, "top_k":top_k, "repetition_penalty":repetition_penalty, "encoder_repetition_penalty":encoder_repetition_penalty, "length_penalty":length_penalty}

                        descr = textgen(prompt, textgen_json)

                        tq.update(1)
            
            tq.close()
        
        if do_export:
            move_the_files(split_video_path, dataset_path, L, max_d, exp_overwrite_dims, exp_w, exp_h, exp_overwrite_fps, exp_fps)

            if do_delete:
                shutil.rmtree(split_video_path)

    # human-friendly image descr regeneration
    def descr_regen_image(image, beam_amount, min_length, max_length):
        load_blip()
        try:
            text = caption_image(image, beam_amount, min_length, max_length)
        except Exception as e:
            logger.error(e)
        finally:
            unload_blip()
        return text

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
    
    def descr_regen(init_path, depth, scene, action, beam_amount, min_length, max_length, textgen_url, textgen_key, max_new_tokens, temperature, top_p, typical_p, top_k, repetition_penalty, encoder_repetition_penalty, length_penalty, master_scene, master_synopsis):
        textgen_json = locals()
        for i in 'init_path,depth,scene,action,beam_amount,min_length,max_length,master_scene,master_synopsis'.split(','):
            textgen_json.pop(i)
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
        if d == max_d:
            image = read_first_frame(action_mp4)
            descr = descr_regen_image(image, beam_amount, min_length, max_length)
        else:
            next_depth_name = os.path.join(depth_name, f'depth_{d+1}')
            next_part_path = os.path.join(next_depth_name, f'part_{action+L*scene}') # `i` cause we want to sample each corresponding *subset*

            # depths > 0 are *guaranteed* to have L videos in their part_j folders
            
            # now sampling each description at the next level
            scenes = ''
            for k in range(L):
                with open(os.path.join(next_part_path, f'subset_{k}.txt'), 'r', encoding='utf-8') as subdescr:
                    scenes += subdescr.read() + '\n'
            
            if d == 0:
                prompt = master_synopsis.replace('%descriptions%', scenes)
            else:
                prompt = master_scene.replace('%descriptions%', scenes)
            
            descr = textgen(prompt, textgen_json)
        
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
                                    master_scene = gr.TextArea(label="Scene", lines=5, interactive=True, value=master_scene_default)
                                with gr.Row(variant='compact'):
                                    master_synopsis = gr.TextArea(label="Synopsis", lines=5, interactive=True, value=master_synopsis_default)
                            with gr.Tab(label='Frame captioning'):
                                gr.Markdown('Frame autocaptioning (BLIP2) settings')
                                gr.Markdown('Uses bisection for more than 1 prompt/division')
                                with gr.Row(variant='compact'):
                                    autocap_frames = gr.Slider(label='Autocaptioned frames', value=2, step=1, interactive=True, minimum=1, maximum=12) # will be populater with L
                                    autocap_beam_amount = gr.Slider(label='Beam amount', value=7, step=1, interactive=True, minimum=1, maximum=30)
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
        descr_regen_btn.click(descr_regen, inputs=[chop_split_path, descr_depth, descr_part, descr_subset, autocap_beam_amount, autocap_min_words, autocap_max_words, textgen_url, textgen_key, textgen_new_words, textgen_temperature, textgen_top_p, textgen_typical_p, textgen_top_k, textgen_repetition_penalty, textgen_encoder_repetition_penalty, textgen_length_penalty, master_scene, master_synopsis], outputs=[descr])
        descr_save_btn.click(write_descr, inputs=[descr, chop_split_path, descr_depth, descr_part, descr_subset], outputs=[])

        # process
        do_btn.click(process_video, inputs=[process_video(do_chop, do_clear, do_caption, do_textgen, do_export, do_delete, chop_whole_vid_path, chop_split_path, chop_trg_path, autocap_beam_amount, autocap_min_words, autocap_max_words, textgen_url, textgen_key, textgen_new_words, textgen_temperature, textgen_top_p, textgen_typical_p, textgen_top_k, textgen_repetition_penalty, textgen_encoder_repetition_penalty, textgen_length_penalty, master_scene, master_synopsis, exp_overwrite_dims, exp_w, exp_h, exp_overwrite_fps, exp_fps)])

    interface.launch(share=args["share"], server_name=args['server_name'], server_port=args['server_port'])
