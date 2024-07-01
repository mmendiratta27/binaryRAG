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

# Text Loading Option
# loader = TextLoader("./rag_wiki.md")

# Using Avengers Endgame as URL
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


# LLM
local_llm = "llama3"
llm = ChatOllama(model=local_llm, temperature=0)

# Search through database for excerpts related to question
retriever = db.as_retriever()
# docs = retriever.invoke(question)

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})

# Prompt Format for Llama3
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    Use the following pieces of retrieved context to determine if the following statement is true or false. You can only respond in one word, unless you do not know the answer in which case answer: 'I don't know.'<|eot_id|><|start_header_id|>user<|end_header_id|>
    Statement: {statement} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["statement", "document"],
)

llm = ChatOllama(model=local_llm, temperature=0)

# Chain
rag_chain = prompt | llm | StrOutputParser()

file1 = open('statements/statements.txt', 'r')
Lines = file1.readlines()

total_start = time.time()

if os.path.exists('statements/results.txt'):
        os.remove('statements/results.txt') #this deletes the file
else:
        print("The file does not exist")#add this to prevent errors


f = open("statements/results.txt", "x")

for statement in Lines:
    # Run
    docs = retriever.invoke(statement)

    # print(docs)
    print("--- Generating Response ---")

    start_time = time.time()
    generation = rag_chain.invoke({"context": docs, "statement": statement})
    print("--- %s seconds ---" % (time.time() - start_time))

    print(f"\nStatement:  {statement}\n")
    print(f"{generation}\n")
    f.write(f"{generation}\n")



print("--- %s Total Time in Seconds ---" % (time.time() - total_start))

# Sources

# For sentence-transformers: https://medium.com/@akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7
# For Ollama: https://ollama.com/download
# General Ref: https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_1_to_4.ipynb