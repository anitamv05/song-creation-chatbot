import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")




from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from gag_generator import generate_gag  # Assuming you have a function for generating gags
from agentic_rags_model import generate_rags  # Assuming you have an agentic rags model

def generate_song(prompt, include_gag=False, include_rags=False):
    llm = OpenAI(temperature=0.7)  # Adjust temperature as needed

    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="Write a song about {prompt}."
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    initial_lyrics = chain.run(prompt)

    if include_gag:
        gag = generate_gag(initial_lyrics)
        initial_lyrics += f"\n{gag}"

    if include_rags:
        rags = generate_rags(initial_lyrics)
        initial_lyrics += f"\n{rags}"

    return initial_lyrics