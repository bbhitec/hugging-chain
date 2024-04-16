'''
    [bbht] hugging-chain
    multi model chaining with HuggingFace models pipelining
    specifically: "describe an image" using the chain:
        -image to text generation (model: salesforce/blip-image-captioning-base)
        -text to speech generation (model: espnet/kan-bayashi_ljspeech_vits)

'''

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline



import requests
import os

load_dotenv(find_dotenv())
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")


########## Image-to-text
def img2text(url):
    img2text_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    text_res = img2text_pipe(url)[0]['generated_text']

    print (text_res)
    return(text_res)






########## Story to Speach
def text2speech(text):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_ACCESS_TOKEN}"}
    payloads = {
        "inputs": text
    }

    print (f"generating audio for: {text}")

    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio_res.flac', 'wb') as file:
        file.write(response.content)


def main():
    # img_url = "https://drive.usercontent.google.com/download?id=1wWpoxNPxiXlqyDIJqT5iPs77IQMGcQj2"
    img_url = "pic1.jpg"
    summary = img2text(img_url)
    # story=text2story(summary)
    text2speech(summary)

if __name__ == "__main__":
    main()
