# To restore poetry env:
# `pip install poetry`
# `poetry shell` `poetry install`

# To change to poetry interpreter:
# In terminal: poetry env info --path
# Copy output to new interpreter: ctr + shift + p > python: select interpreter & paste in


import os
import s3fs

from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = 'all-mpnet-base-v2'
embeddings_folder = 'all-mpnet'

#inference_model = "TheBloke/orca_mini_3B-GGML"
#inference_model = 'bardsai/jaskier-7b-dpo-v5.6'
#inference_model='pankajmathur/orca_mini_3b'
#inference_model= 'TinyLlama/TinyLlama_v1.1'
#inference_model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
#inference_folder = "tinylama"

def download_inference_model(base_model_name: str, s3, s3_model_path: str):
    """Downloads chosen huggingface model to cache_dir"""
    model = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer = AutoModel.from_pretrained(base_model_name)

    # Save the model and tokenizer directly to the S3 bucket

    temp_path = f'{os.getcwd()}/inference_model'
    model.save_pretrained(f'{temp_path}/model')
    tokenizer.save_pretrained(f'{temp_path}/tokenizer' )

    # Upload saved files to S3
    for filename in os.listdir(f'{temp_path}/model'):
        local_path = os.path.join(f'{temp_path}/model', filename)
        s3_path = f'{s3_model_path}/{filename}'
        print(s3_path)
        with open(local_path, 'rb') as f:
            s3.put(local_path, s3_path) 

    for filename in os.listdir(f'{temp_path}/tokenizer'):
        local_path = os.path.join(f'{temp_path}/tokenizer', filename)
        s3_path = f'{s3_model_path}/{filename}'
        print(s3_path)
        with open(local_path, 'rb') as f:
            s3.put(local_path, s3_path)

def download_embeddings_model(embeddings_model: str, s3, s3_embeddings_path:str):
    temp_path = f'{os.getcwd()}/embeddings_model'
    embeddings_model = HuggingFaceEmbeddings(model_name = embeddings_model, cache_folder = temp_path)
    
    for filename in os.listdir(temp_path):
        local_path = os.path.join(temp_path, filename)
        s3_path = f'{s3_model_path}/{filename}'
        with open(local_path, 'rb') as f:
            s3.put(local_path, s3_path)

if __name__ == "__main__":

    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    print(S3_ENDPOINT_URL)
    s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL}) # do I need this?

    bucket = "sjentoft"
    s3_model_path = f'{bucket}/inference-models/{inference_folder}'
    model_path ="inference_model"
    temp_path = f'{os.getcwd()}/inference_model'

    # download inference model locally and move - add in cleanup later
    download_inference_model(inference_model, s3, s3_model_path)
    
    # download embeddings model
    s3_embeddings_path = f'{bucket}/embeddings-models/{embeddings_folder}'
    download_embeddings_model(embeddings_model, s3, s3_embeddings_path)

    print("models saved!")



