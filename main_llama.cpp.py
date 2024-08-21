from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llamaCpp_path = os.getenv('LOCAL_LLAMACPP_PATH')

with open('prompts/prompt_formatted.txt') as f:
    template = f.read()

prompt_template = PromptTemplate.from_template(template)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=llamaCpp_path,
    temperature=0,
    max_tokens=300,
    top_p=1,
    callback_manager=callback_manager,
    echo=True,
    # verbose=True,  # Verbose is required to pass to the callback manager
)

prompt = prompt_template.format(
    user_message="the house is green"
)
print(prompt)

llm.invoke(prompt)