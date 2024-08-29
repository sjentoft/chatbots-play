# Simple case for testing a qa RAG bot


# Import pacakges
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
model_path = "inference-models/tinylama"

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


if __name__ == "__main__":
    # Set up vectorstore
    s = time.time()
    docs = create_db.split_html(bucket, prefix)
    vectorstore = create_db.create_vectorstore(docs)
    t = time.time() - s
    print(f"vectore store done in {t} seconds")

    s = time.time()
    llm = get_llm(model_path)
    t = time.time() - s
    print(f"llm loaded in {t} seconds")

    # Get q's - replace with those from csv file. 
    qs_vec = ["What is Dapla?", "What is a PAT?"]

    template = """Use the following pieces of context to answer the users question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        ALWAYS return a "SOURCES" part in your answer.

        Context: {context}
        Question: {question}
        """

    prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    memory = SimpleMemory()

    s = time.time()
    qa_chain = ConversationalRetrievalChain.from_llm(llm, 
            retriever=vectorstore.as_retriever(), 
            memory=memory,
            verbose=True)
    
    answers = []
    for q in qs_vec:
        response = qa_chain({'question': q, "chat_history": []})
        answers.append(response["answer"])
    print(answers)

    print(f'Response took: {time.time() - s} seconds')
