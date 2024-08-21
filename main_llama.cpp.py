""""
This script enables the calling of a LLM from local files, instead of API calling
"""
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llamaCpp_path = os.getenv('LOCAL_LLAMACPP_PATH')

with open('prompts/prompt_formatted.txt') as f:
    template = f.read()

prompt_template = PromptTemplate.from_template(template)

# Build a llm lc model, based from local LLama-Cpp
llm = LlamaCpp(
    model_path=llamaCpp_path,
    temperature=0,
    max_tokens=300,
    top_p=1,
    echo=True, # enables the add of special tokens in the output 
)

# Define lc chain
chain = prompt_template | llm

response = chain.invoke(
    {
        'user_message': 'the house is green'
    }
)

print(response)