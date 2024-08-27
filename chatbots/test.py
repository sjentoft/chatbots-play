
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import transformers
import s3fs

import os

import create_db
inference_model='TinyLlama/TinyLlama_v1.1'



dato = "2024-08-16"

bucket = "sjentoft"
prefix = f"db-files/dapla-manual/{dato}"

docs = create_db.split_html(bucket, prefix)
    
vectorestore = create_db.create_vectorstore(docs)
print("vectore store done")


def get_llm(model_path):

# S3 bucket and file paths
    s3_bucket = 'sjentoft'
    model_path = 'inference-models'

    # Initialize s3fs
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    local_dir = f'{os.getcwd()}/tmp/model'

    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Copy the model and tokenizer from S3 to the local directory
    fs.get(f'{s3_bucket}/{model_path}', local_dir, recursive=True)
    #fs.get(f'{s3_bucket}/{tokenizer_path}', f'local_dir/{tokenizer_path}', recursive=True)

    # Load the model and tokenizer from the local directory
    model = AutoModelForCausalLM.from_pretrained(f'{local_dir}/{model_path}')
    tokenizer = AutoTokenizer.from_pretrained(f'{local_dir}/{model_path}')

    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Wrap the pipeline in a langchain LLM
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    return llm

model_path = "s3/sjentoft/inference-models/"

llm = get_llm(model_path)
print("llm loaded")

def chatty_bot(llm, vectorstore):
    # Set up question/answer scheme
    qa = ConversationalRetrievalChain.from_llm(
        llm, 
        vectorstore.as_retriever(), 
        return_source_documents=True
    )

    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit"]:
            break

        # Get result from QA chain
        response = qa({'question': user_message, 'chat_history': []})
        
        # Print the response
        print("Bot:", response["answer"])

# Example usage
chatty_bot(llm, vectorestore)


