import torch, os, logging, sys, shutil, time
# Don't forget to pip install videollava!
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from typing import Union
import uvicorn, fire
from fastapi import FastAPI, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler(f'video_llava_{time.strftime("%Y%m%d%H%M%S")}.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(stdout_handler)

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)

def setup():
    disable_torch_init()
    cache_dir = 'cache_dir'
    device = 'cuda'
    load_4bit, load_8bit = True, False
    model_path = 'LanguageBind/Video-LLaVA-7B'

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']

    return tokenizer, model, video_processor

def main(port=7681):
    app = FastAPI()

    logger.info('Setting up the server for VideoLLAVA')
    tokenizer, model, video_processor = setup()
    logger.info('Models loaded successfully')

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
        )

    @app.post("/describe")
    def describe_uploaded_video(prompt: str, video_file: UploadFile):
        logger.debug(f'Incoming video {video_file.filename}!')
        filename = "temp."+str(video_file.filename).split('.')[-1]
        try:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(video_file.file, f)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            video_file.file.close()
        
        video = filename #'videollava/serve/examples/sample_demo_1.mp4'
        inp = prompt #'Why is this video funny?'

        # NOTE: conv gets reset every request
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles

        video_tensor = video_processor(video, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        logger.debug(f"{roles[1]}: {inp}")
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=100,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        reply = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        logger.debug(f'Description is: {reply}')

        return {"message": reply}

    uvicorn.run(app, port=port)

if __name__ == '__main__':
    fire.Fire(main)
