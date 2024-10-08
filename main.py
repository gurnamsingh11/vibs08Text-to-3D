from gradio_client import Client
import requests
import os
from typing import Union
from fastapi import FastAPI, Form
import json
import boto3
import json
import base64
from PIL import Image
from functools import partial
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust as needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

app.add_middleware(HTTPSRedirectMiddleware)


client = Client("vibs08/flash-sd3-new", hf_token="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")

url = 'https://vibs08-image-3d-fastapi.hf.space/process_image/'

ACCESS = os.getenv("ACCESS")
SECRET = os.getenv("SECRET")
bedrock = boto3.client(service_name='bedrock', aws_access_key_id = ACCESS, aws_secret_access_key = SECRET, region_name='us-east-1')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', aws_access_key_id = ACCESS, aws_secret_access_key = SECRET, region_name='us-east-1')



def gen_pos_prompt(text):
  instruction = f'''Your task is to create a positive prompt for image generation under 28 words.

    Objective: Generate images that prioritize structural integrity and accurate shapes. The focus should be on the correct form and basic contours of objects, with minimal concern for colors.
    
    Guidelines:
    
    Complex Objects (e.g., animals, vehicles, Machines, Fictional Characters, Fantasy and Mythical Creatures, Historical or Cultural Artifacts, Humanoid Figures,Buildings and Architecture): For these, the image should resemble a toy object, emphasizing the correct shape and structure while minimizing details and color complexity.
    
    Example Input: A sports bike
    Example Positive Prompt: Simple sports bike with accurate shape and structure, minimal details, digital painting, concept art style, basic contours, soft lighting, clean lines, neutral or muted colors, toy-like appearance, low contrast.
    
    Example Input: A lion
    Example Positive Prompt: Toy-like depiction of a lion with a focus on structural accuracy, minimal details, digital painting, concept art style, basic contours, soft lighting, clean lines, neutral or muted colors, simplified features, low contrast.

    Input: The Spiderman with Wolverine Claws
    Positive Prompt: Toy-like depiction of Spiderman with Wolverine claws, emphasizing structural accuracy with minimal details, digital painting, concept art style, basic contours, soft lighting, clean lines, neutral or muted colors, simplified features, low contrast.
    
    Simple Objects (e.g., a tennis ball): For these, the prompt should specify a realistic depiction, focusing on the accurate shape and structure.
    
    Example Input: A tennis ball
    Example Positive Prompt: photorealistic, uhd, high resolution, high quality, highly detailed; A tennis ball
    
    Prompt Structure:
    
    Subject: Clearly describe the object and its essential shape and structure.
    Medium: Specify the art style (e.g., digital painting, concept art).
    Style: Include relevant style terms (e.g., simplified, toy-like for complex objects; realistic for simple objects).
    Resolution: Mention resolution if necessary (e.g., basic resolution).
    Lighting: Indicate the type of lighting (e.g., soft lighting).
    Color: Use neutral or muted colors with minimal emphasis on color details.
    Additional Details: Keep additional details minimal.


    Input: {text}
    Positive Prompt: 
    '''

  body = json.dumps({'inputText': instruction,
                     'textGenerationConfig': {'temperature': 0, 'topP': 0.01, 'maxTokenCount':1024}})
  response = bedrock_runtime.invoke_model(body=body, modelId='amazon.titan-text-express-v1')
  pos_prompt = json.loads(response.get('body').read())['results'][0]['outputText']
  return pos_prompt

def text2img(promptt):

    result = client.predict(
        prompt=promptt,  
        seed=0,
        randomize_seed=True,
        guidance_scale=1,
        num_inference_steps=4,
        negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW, bad text",
        api_name="/infer"
    )
    

    return result

def aws_bedrock(pos_prompt,output_path='output_image.png'):
  new_prompt = gen_pos_prompt(pos_prompt)
  print(new_prompt)
  neg_prompt = '''deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW, bad text'''
    
  parameters = {
      'taskType': 'TEXT_IMAGE',
      'textToImageParams': {'text': new_prompt,
                            'negativeText': neg_prompt},
      'imageGenerationConfig': {"cfgScale":8,
                                "seed":0,
                                "width":1024,
                                "height":1024,
                                "numberOfImages":1
                                }
  }
  request_body = json.dumps(parameters)
  response = bedrock_runtime.invoke_model(body=request_body, modelId='amazon.titan-image-generator-v1')
  response_body = json.loads(response.get('body').read())
  base64_image_data = base64.b64decode(response_body['images'][0])

 
  with Image.open(io.BytesIO(base64_image_data)) as img:
        img.save(output_path)
    
  return output_path


def stability(image):
  url = "https://vibs08-stable-fast-3d.hf.space/process-image/?foreground_ratio=0.85"


  files = {
      "file": (image, open(image, "rb"), "image/png"),
  }

  response = requests.post(url, files=files)

  return response.json()


def three_d(prompt, seed, fr, mc, auth, text=None):
    new_prompt = gen_pos_prompt(prompt)

    results = {}
    
    # Try running text2img
    try:
      text2img_file_path = text2img(new_prompt)
      with open(text2img_file_path, 'rb') as image_file:
        files = {
            'file': (text2img_file_path, image_file, 'image/png')
            }
        headers = {
                'accept': 'application/json'
                }
        payload = {
                    'seed': seed,
                    'enhance_image': False,
                    'do_remove_background': True,
                    'foreground_ratio': fr,
                    'mc_resolution': mc,
                    'auth': auth,
                    'text_prompt': text
                    }
        response = requests.post(url, headers=headers, files=files,data=payload)
        result = response.json()
        results["text2img_img_path"] = result.get("img_path")
        results["text2img_obj_path"] = result.get("obj_path")
        results["text2img_glb_path"] = result.get("glb_path")
    except Exception as e:
      results["text2img_error"] = str(e)
    try:
      text2img_file_path = text2img(new_prompt)
      new_response = stability(text2img_file_path)
      results["stabilityai_glb_path"] = new_response.get("glb2_path")
    except Exception as e:
      results["stability__error"] = str(e)

    # Try running aws_bedrock
    try:
      aws_bedrock_file_path = aws_bedrock(new_prompt)
      with open(aws_bedrock_file_path, 'rb') as image_file:
            files = {
                'file': (aws_bedrock_file_path, image_file, 'image/png')
            }

            headers = {
                'accept': 'application/json'
            }
            payload = {
                'seed': seed,
                'enhance_image': False,
                'do_remove_background': True,
                'foreground_ratio': fr,
                'mc_resolution': mc,
                'auth': auth,
                'text_prompt': text
                }
            response = requests.post(url, headers=headers, files=files,data=payload)
            result = response.json()
            results["aws_bedrock_img_path"] = result.get("img_path")
            results["aws_bedrock_obj_path"] = result.get("obj_path")
            results["aws_bedrock_glb_path"] = result.get("glb_path")
    except Exception as e:
      results["aws_bedrock_error"] = str(e)

    try:
      aws_bedrock_file_path = aws_bedrock(new_prompt)
      new_response = stability(aws_bedrock_file_path)
      results["aws_stability_glb_path"] = new_response.get("glb2_path")
    except Exception as e:
      results["aws_stability_error"] = str(e)

    return results   

@app.post("/process_text/")
async def process_text(
    text_prompt: str = Form(...),
    seed: int = Form(...),
    foreground_ratio: float = Form(...),
    mc_resolution: int = Form(...),
    auth: str = Form(...)
    #use_aws: bool = Form(...)
):
    if auth == "userName":
        return three_d(text_prompt, seed, foreground_ratio, mc_resolution, auth)
    else:
        return {"Authentication": "Failed"}

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=7860)
