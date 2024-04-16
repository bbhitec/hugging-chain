'''
    [bbht] hugging-chain
    multi model chaining with HuggingFace models pipelining

'''

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())

########## Image-to-text
def img2text(url):
    img2text_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    text_res = img2text_pipe(url)

    print (text_res[0]['generated_text'])
    return(text_res)




########## Story generation (via LLM)



########## Story to Speach



def main():
    img_url = "https://drive.usercontent.google.com/download?id=1wWpoxNPxiXlqyDIJqT5iPs77IQMGcQj2"
    img2text(img_url)

if __name__ == "__main__":
    main()
