from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def generate_song(prompt):
    llm = OpenAI(temperature=0.7)  # Adjust temperature as needed

    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="Write a song about {prompt}."
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    return chain.run(prompt)