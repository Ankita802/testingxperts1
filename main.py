import os 
from constants import openai_key 
from langchain.llms import GooglePalm
from langchain import PromptTemplate 
from langchain.chains import LLMChain
import streamlit as st 
from constants import openai_key


# os.environ['OPENAI_API_KEY'] = openai_key

api_key = "AIzaSyBkIxznvzMOl84OIaSEU6YOhQXNNlmuDAg" 

# streamlit framework 

st.title('Langchain demo with GooglePalm')
input = st.text_input("Search the topic you want")

## GooglePalm LLMs

llm = GooglePalm(google_api_key = api_key,temperature = 0.3)


if input :
    st.write(llm(input))


