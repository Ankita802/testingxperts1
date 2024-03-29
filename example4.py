from langchain.llms import GooglePalm
import os

api_key="AIzaSyBkIxznvzMOl84OIaSEU6YOhQXNNlmuDAg"

llm=GooglePalm(google_api_key=api_key, temperature=0.7)


os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

from langchain_community.llms import HuggingFaceHub
 
llm = HuggingFaceHub(
    repo_id="salony/User_story",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,  
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
)



