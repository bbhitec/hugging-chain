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

import streamlit as st  # quick ui lib

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



########## main driver
def main():

    # defining ui
    st.set_page_config(page_title="A Joke about your pic?", page_icon="😁")
    st.header("AI Image-to-Joke Generator")

    img_uploaded = st.file_uploader("Give me an image...", type="jpg")
    if img_uploaded:
        # save the given image
        img_data = img_uploaded.getvalue()
        with open(img_uploaded.name, "wb") as file:
            file.write(img_data)

        # show it
        st.image(img_uploaded, caption="Your Image", use_column_width=True)

        # run and the models
        summary = img2text(img_uploaded.name)
        joke=text2story(summary)
        text2speech(joke)

        # present results
        with st.expander("Whats I see in the image:"):
            st.write(summary)
        with st.expander("Here's a joke about it..."):
            st.write(joke)
        st.audio("audio_res.flac")

if __name__ == "__main__":
    main()
