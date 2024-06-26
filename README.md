# Hugging Face Multi-model Utilization

Multi model chaining with HuggingFace transformers pipelines and LangChain interfacing. </br>
Study case: Pick-a-Joke app: "say a quick joke about the given image" using the tools:
- Image to text generation (model: salesforce/blip-image-captioning-base)
- Templated prompt text generation using LangChain with OpenAI models
- Text to speech generation (model: espnet/kan-bayashi_ljspeech_vits)
- Pipelining HuggingFace models for local use
- Using HuggingHace Inference APIs to use models remotely via HTTP requests
- Quick app UI with Streamlit UI
- Deploying demo app to HuggingFace Spaces

</br>

![](https://shields.io/badge/-python-ffe600?logo=python)
![](https://shields.io/badge/-pytorch-4377cb?logo=pytorch)
![](https://shields.io/badge/-huggingface-4377cb?logo=huggingface)
![](https://shields.io/badge/-langchain-4377cb?logo=langchain)
![](https://shields.io/badge/-streamlit-4377cb?logo=streamlit)


## Usage/How to run
Pull and install required packages if needed (or see the [`demo`](https://huggingface.co/spaces/vnik/pic-a-joke)):

```python
pip install -r requirements.txt
```

_*Additional dependencies may be required_</br>
_*Additional environment variables may be needed_</br>
_*Virtual env recommended_</br>

</br>

##### [vnik]
