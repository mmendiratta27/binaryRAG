from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import time
import os
import subprocess

num_statements = 20
num_prompts = 5
num_models = 3

# Using Avengers for URLs
urls = ["https://en.wikipedia.org/wiki/The_Avengers_(2012_film)", "https://en.wikipedia.org/wiki/Avengers:_Age_of_Ultron", "https://en.wikipedia.org/wiki/Avengers:_Endgame", "https://en.wikipedia.org/wiki/Avengers:_Infinity_War"]
loader = WebBaseLoader(web_paths=urls)

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

start_time = time.time()
db = FAISS.from_documents(docs, embeddings)

print("--- Embedding Documents ---")
print("--- %s seconds ---" % (time.time() - start_time))



# Search through database for excerpts related to question
retriever = db.as_retriever()
# docs = retriever.invoke(question)

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})

# Prompt Format for Llama3
prompt_temp = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    {prompt_eng}<|eot_id|><|start_header_id|>user<|end_header_id|>
    Statement: {statement} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["statement", "document"],
)


file1 = open('statements.txt', 'r')
Statements = file1.readlines()

file1 = open('prompts.txt', 'r')
Prompts = file1.readlines()

# output contains generation from LLM
if os.path.exists('output.txt'):
        os.remove('output.txt')

output = open("output.txt", "x")

# results contains final output with timings and accuracy
if os.path.exists('results.txt'):
        os.remove('results.txt')

results = open("results.txt", "x")

# LLM Otions
# llama3 (8b)
# phi3:mini (3b)
# llama3:70b

llm_models = ["phi3:mini", "llama3", "llama3:70b"]

expected = ["False", "False", "False", "False", "False", "False", "False", "False", "False", "False", "False", "True", "True", "True", "True", "True", "True", "True", "True", "True", "True"]
counter = 0

num_correct_prompt = 0
num_correct_model = 0
# , "llama3", "llama3:70b"

for j, model in enumerate(llm_models):
    
    llm = ChatOllama(model=model, temperature=0)
    # Chain
    rag_chain = prompt_temp | llm | StrOutputParser()
    output.write(f"Model {model}\n-------------------\n")
    results.write(f"Model {model}\n-------------------\n")
    total_start = time.time()
    for i, prompt in enumerate(Prompts, start=1):
        p_time = time.time()
        
        output.write(f"Prompt #{i}:\n")


        for statement in Statements:
            # Run
            docs = retriever.invoke(statement)
            generation = rag_chain.invoke({"prompt_eng": prompt, "context": docs, "statement": statement})
            
            if expected[counter] in generation:
                num_correct_prompt += 1
                num_correct_model += 1
            counter += 1
            output.write(f"{generation}\n")

        print(f"--- Prompt #{i}: {(time.time() - p_time)} seconds ---")
        results.write(f"--- Prompt #{i}: {(time.time() - p_time)} seconds ---\n")

        print(f"--- Percentage Correct: {100*num_correct_prompt/num_statements} ---")
        results.write(f"--- Percentage Correct: {100*num_correct_prompt/num_statements} ---\n")
        
        # reset prompt accuracy count
        num_correct_prompt = 0
        counter = 0
    
    print(f"\n--- {model}: {(time.time() - total_start)} seconds & {100*num_correct_model/(num_statements*num_prompts)}% Correct---\n")
    results.write(f"\n--- {model}: {(time.time() - total_start)} seconds & {100*num_correct_model/(num_statements*num_prompts)}% Correct---\n")
    
    # reset model accuracy count
    num_correct_model = 0


# Sources

# For sentence-transformers: https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7
# For Ollama: https://ollama.com/download
# General Ref: https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_1_to_4.ipynb