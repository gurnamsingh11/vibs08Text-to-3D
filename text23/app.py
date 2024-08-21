from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os
import boto3
import json
import shlex
import subprocess
import tempfile
import time
import base64
import gradio as gr
import numpy as np
import rembg
import spaces
import torch
from PIL import Image
from functools import partial
import io
import datetime

app = FastAPI()


subprocess.run(shlex.split('pip install wheel/torchmcubes-0.1.0-cp310-cp310-linux_x86_64.whl'))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation


HEADER = """FRAME AI"""


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# torch.cuda.synchronize()


model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(131072)
model.to(device)

rembg_session = rembg.new_session()
ACCESS = os.getenv("ACCESS")
SECRET = os.getenv("SECRET")
bedrock = boto3.client(service_name='bedrock', aws_access_key_id = ACCESS, aws_secret_access_key = SECRET, region_name='us-east-1')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', aws_access_key_id = ACCESS, aws_secret_access_key = SECRET, region_name='us-east-1')

def upload_file_to_s3(file_path, bucket_name, object_name=None):
    s3_client = boto3.client('s3',aws_access_key_id = ACCESS, aws_secret_access_key = SECRET, region_name='us-east-1')
    
    if object_name is None:
        object_name = file_path
    
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return False
    except NoCredentialsError:
        print("Credentials not available.")
        return False
    except PartialCredentialsError:
        print("Incomplete credentials provided.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    
    print(f"File {file_path} uploaded successfully to {bucket_name}/{object_name}.")
    return True


def gen_pos_prompt(text):
  instruction = f'''Your task is to create a positive prompt for image generation.

    Objective: Generate images that prioritize structural integrity and accurate shapes. The focus should be on the correct form and basic contours of objects, with minimal concern for colors.
    
    Guidelines:
    
    Complex Objects (e.g., animals, vehicles): For these, the image should resemble a toy object, emphasizing the correct shape and structure while minimizing details and color complexity.
    
    Example Input: A sports bike
    Example Positive Prompt: Simple sports bike with accurate shape and structure, minimal details, digital painting, concept art style, basic contours, soft lighting, clean lines, neutral or muted colors, toy-like appearance, low contrast.
    
    Example Input: A lion
    Example Positive Prompt: Toy-like depiction of a lion with a focus on structural accuracy, minimal details, digital painting, concept art style, basic contours, soft lighting, clean lines, neutral or muted colors, simplified features, low contrast.
    
    Simple Objects (e.g., a tennis ball): For these, the prompt should specify a realistic depiction, focusing on the accurate shape and structure.
    
    Example Input: A tennis ball
    Example Positive Prompt: Realistic depiction of a tennis ball with accurate shape and texture, digital painting, clean lines, minimal additional details, soft lighting, neutral or muted colors, focus on structural integrity.
    
    Prompt Structure:
    
    Subject: Clearly describe the object and its essential shape and structure.
    Medium: Specify the art style (e.g., digital painting, concept art).
    Style: Include relevant style terms (e.g., simplified, toy-like for complex objects; realistic for simple objects).
    Resolution: Mention resolution if necessary (e.g., basic resolution).
    Lighting: Indicate the type of lighting (e.g., soft lighting).
    Color: Use neutral or muted colors with minimal emphasis on color details.
    Additional Details: Keep additional details minimal or specify if not desired.


    Input: {text}
    Positive Prompt: 
    '''

  body = json.dumps({'inputText': instruction,
                     'textGenerationConfig': {'temperature': 0.1, 'topP': 0.01, 'maxTokenCount':512}})
  response = bedrock_runtime.invoke_model(body=body, modelId='amazon.titan-text-express-v1')
  pos_prompt = json.loads(response.get('body').read())['results'][0]['outputText']
  return pos_prompt

def generate_image_from_text(pos_prompt, seed):
  new_prompt = gen_pos_prompt(pos_prompt)
  print(new_prompt)
  neg_prompt = '''Detailed, complex textures, intricate patterns, realistic lighting, high contrast, reflections, fuzzy surface, realistic proportions, photographic quality, vibrant colors, detailed background, shadows, disfigured, deformed, ugly, multiple, duplicate.'''
  neg_prompt = '''Complex textures, intricate patterns, realistic lighting, high contrast, reflections, fuzzy surface, photographic quality, vibrant colors, detailed background, shadows, disfigured, deformed, ugly, multiple, duplicate.'''

    
  parameters = {
      'taskType': 'TEXT_IMAGE',
      'textToImageParams': {'text': new_prompt,
                            'negativeText': neg_prompt},
      'imageGenerationConfig': {"cfgScale":8,
                                "seed":int(seed),
                                "width":512,
                                "height":512,
                                "numberOfImages":1
                                }
  }
  request_body = json.dumps(parameters)
  response = bedrock_runtime.invoke_model(body=request_body, modelId='amazon.titan-image-generator-v1')
  response_body = json.loads(response.get('body').read())
  base64_image_data = base64.b64decode(response_body['images'][0])

  return Image.open(io.BytesIO(base64_image_data))

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        torch.cuda.synchronize()  # Ensure previous CUDA operations are complete
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        return image

    if do_remove_background:
        torch.cuda.synchronize()
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
        
        torch.cuda.synchronize()
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    torch.cuda.synchronize()  # Wait for all CUDA operations to complete
    torch.cuda.empty_cache()
    return image



# @spaces.GPU
def generate(image, mc_resolution, formats=["obj", "glb"]):
    torch.cuda.synchronize()
    scene_codes = model(image, device=device)
    torch.cuda.synchronize()
    mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]
    torch.cuda.synchronize()
    mesh = to_gradio_3d_orientation(mesh)
    torch.cuda.synchronize()
    
    mesh_path_glb = tempfile.NamedTemporaryFile(suffix=f".glb", delete=False)
    torch.cuda.synchronize()
    mesh.export(mesh_path_glb.name)
    torch.cuda.synchronize()
    
    mesh_path_obj = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False)
    torch.cuda.synchronize()
    mesh.apply_scale([-1, 1, 1])
    mesh.export(mesh_path_obj.name)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return mesh_path_obj.name, mesh_path_glb.name

    

def run_example(text_prompt,seed ,do_remove_background, foreground_ratio, mc_resolution):
    image_pil = generate_image_from_text(text_prompt, seed)

    preprocessed = preprocess(image_pil, do_remove_background, foreground_ratio)
    
    mesh_name_obj, mesh_name_glb = generate(preprocessed, mc_resolution, ["obj", "glb"])
    
    return preprocessed, mesh_name_obj, mesh_name_glb


@app.post("/process_text/")
async def process_image(
    text_prompt: str = Form(...),
    seed: int = Form(...),
    do_remove_background: bool = Form(...),
    foreground_ratio: float = Form(...),
    mc_resolution: int = Form(...),
    auth: str = Form(...)
):
    
    if auth == os.getenv("AUTHORIZE"):

        preprocessed, mesh_name_obj, mesh_name_glb = run_example(text_prompt,seed ,do_remove_background, foreground_ratio, mc_resolution)
        # preprocessed = preprocess(image_pil, do_remove_background, foreground_ratio)
        # mesh_name_obj, mesh_name_glb = generate(preprocessed, mc_resolution)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        object_name = f'object_{timestamp}_1.obj'
        object_name_2 = f'object_{timestamp}_2.glb'
    
        if upload_file_to_s3(mesh_name_obj, 'framebucket3d',object_name) and upload_file_to_s3(mesh_name_glb, 'framebucket3d',object_name_2):
    
            return {
                "obj_path": f"https://framebucket3d.s3.amazonaws.com/{object_name}",
                "glb_path": f"https://framebucket3d.s3.amazonaws.com/{object_name_2}"

            }
    
        else:
            return {"Internal Server Error": False}
    else:
        return {"Authentication":"Failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)