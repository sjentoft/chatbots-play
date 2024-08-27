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


#inference_model = "TheBloke/orca_mini_3B-GGML"
#inference_model = 'bardsai/jaskier-7b-dpo-v5.6'
#inference_model='pankajmathur/orca_mini_3b'
inference_model='TinyLlama/TinyLlama_v1.1'

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

    

if __name__ == "__main__":

    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    print(S3_ENDPOINT_URL)
    s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL}) # do I need this?

    bucket = "sjentoft"
    s3_model_path = f'{bucket}/inference-models'
    model_path ="inference_model"
    temp_path = f'{os.getcwd()}/inference_model'


    #s3.touch(s3_model_path)

    # download locally - fix later
    download_inference_model(inference_model, s3, s3_model_path)

    print("models saved!")



