'''
    [bbht] hugging-chain
    multi model chaining with HuggingFace models pipelining
    specifically: "describe an image" using the chain:
        -image to text generation (model: salesforce/blip-image-captioning-base)
        -templated prompt text generation using LangChain with OpenAI
        -text to speech generation (model: espnet/kan-bayashi_ljspeech_vits)

'''

from dotenv import load_dotenv
from transformers import pipeline

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import requests
import os

load_dotenv()   # take environment variables from .env
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


########## Image-to-text
def img2text(url):
    img2text_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=20)

    text_res = img2text_pipe(url)[0]['generated_text']

    print (text_res)
    return(text_res)


########## Joke generation
def text2story(summary):

    model = ChatOpenAI(model="gpt-4")

    prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    story = chain.invoke({"topic": {summary}})

    print(story)
    return(story)


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
        print("file outputted!")




def main():
    img_url = "pic1.jpg"
    summary = img2text(img_url)
    story=text2story(summary)
    text2speech(story)

if __name__ == "__main__":
    main()
