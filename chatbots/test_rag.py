# Simple case for testing a qa RAG bot

# From Benedikt
from vllm import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import VLLM

# Import packges
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import transformers
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
#from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import SimpleMemory

import s3fs
import os
import time

import create_db

# Set up parameters
inference_model='TinyLlama/TinyLlama_v1.1'
dato = "2024-08-16" # For dp manual scraping version
bucket = "sjentoft"
prefix = f"db-files/dapla-manual/{dato}"
model_path = "inference-models/mistral"
local_path = "inference_model/mistral"

# Get model from S3 bucket
def get_llm(model_path):
    """
    Function to get model from S3 bucket ond save locally for reading in.
    """
    # S3 bucket and file paths
    s3_bucket = 'sjentoft'

    # Initialize s3fs
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    local_dir = f'{os.getcwd()}/tmp/model'

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

        # Copy the model and tokenizer from S3 to the local directory
        fs.get(f'{s3_bucket}/{model_path}', local_dir, recursive=True)

    # Load the model and tokenizer from the local directory
    model = AutoModelForCausalLM.from_pretrained(f'{local_dir}/{model_path}')
    tokenizer = AutoTokenizer.from_pretrained(f'{local_dir}/{model_path}')

    #qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device = 0)
    #llm = HuggingFacePipeline(pipeline=qa_pipeline)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

def get_llm_new(model_path, local_path):
    """
    Function to get model from S3 bucket ond save locally for reading in.
    """
    # S3 bucket and file paths
    s3_bucket = 'sjentoft'

    # Initialize s3fs
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    local_dir = f'/home/onyxia/work/chatbots-play/{local_path}'

    # Create the local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

        # Copy the model and tokenizer from S3 to the local directory
        fs.get(
            f'{s3_bucket}/{model_path}',
            '/home/onyxia/work/chatbots-play/inference_model', 
            recursive=True)
        print(f"Model copied to local folder: {local_dir}")

    print(f"Loading model from: {local_dir}")

    llm = VLLM(
        model=local_dir, 
        max_model_len=8000, 
        gpu_memory_utilization=0.9,
        #tensor_parallel_size=1,
        trust_remote_code=True,
        #enforce_eager=True
        #download_dir='models'
    )
    # Load the model and tokenizer from the local directory
    model = AutoModelForCausalLM.from_pretrained(f'{local_dir}/{model_path}')
    tokenizer = AutoTokenizer.from_pretrained(f'{local_dir}/{model_path}')

    #qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device = 0)
    #llm = HuggingFacePipeline(pipeline=qa_pipeline)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048)
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm
    return llm

def get_template():
    template = """Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.

    Context: {context}
    Question: {question}

    Helpful answer:
    """
    return template

def get_prompt():
    template = get_template()
    prompt = PromptTemplate(
        input_variables=["context", "human_input"], template=template
    )
    return prompt

if __name__ == "__main__":
    # Set up vectorstore
    s = time.time()
    docs = create_db.split_html(bucket, prefix)
    vectorstore = create_db.create_vectorstore(docs)
    t = time.time() - s
    print(f"vectore store done in {t} seconds")

    s = time.time()
    llm = get_llm_new(model_path, local_path)
    t = time.time() - s
    print(f"llm loaded in {t} seconds")

    # Get q's - replace with those from csv file. 
    qs_vec = ["What is Dapla?", "What is a PAT?"]

    prompt = get_prompt()

    rag_chain = (
        {"context": vectorstore.as_retriver(), "question":RunnablePassthrough()}
        |prompt
        |llm
        |StrOutputParser()
    )
    
    s = time.time()

    answers = []
    for q in qs_vec:
        response = rag_chain.invoke(q)
        answers.append(response)
    print(answers)

    print(f'Response took: {time.time() - s} seconds')
