# # Create Chroma DB for dapla chatbot

# To restore poetry env:
# `pip install poetry`
# `poetry shell` `poetry install`

# To change to poetry interpreter:
# In terminal: poetry env info --path
# Copy output to new interpreter: ctr + shift + p > python: select interpreter & paste in

import yaml #poetry add pyyaml
import pickle
import os
import getpass
from datetime import date
import s3fs

from sentence_transformers import SentenceTransformer #poetry add sentence-transformers
from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.document_loaders import S3DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from google.cloud import storage #poetry add google-cloud-storage
from langchain_core.document_loaders import BaseLoader
from langchain.schema import Document

from bs4 import BeautifulSoup

from langchain_huggingface.embeddings import HuggingFaceEmbeddings


dato = "2024-08-16"

def remove_single_word_lines(text):
    # Split the text into lines
    lines = text.splitlines()

    # Filter out lines that contain only one word
    filtered_lines = [line for line in lines if len(line.split()) > 1]

    # Join the filtered lines back into a single string
    return '\n'.join(filtered_lines)


def split_html(bucket, prefix, chunk_size = 512):
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    txt_loader = S3DirectoryLoader(bucket, prefix)

    documents = []

    try:
        raw_documents = txt_loader.load()
        # Extract the content from the loaded documents
        for key, content in raw_documents.items():
            text_content = content.decode('utf-8') if isinstance(content, bytes) else content

            # Parse HTML and extract text
            soup = BeautifulSoup(text_content, 'html.parser')
            clean_text = soup.get_text(separator='\n', strip=True)

            # Remove single-word lines
            clean_text = remove_single_word_lines(clean_text)

            # Normalize whitespace
            clean_text = ' '.join(clean_text.split())
            
            # Wrap the cleaned text in a Document object
            documents.append(Document(page_content=clean_text))

    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return []

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    return chunks


def get_embedding_model(model_name):
    file_name = f'./local_model/{model_name}.pkl'
    with open(file_name, 'rb') as file:
        mod = pickle.load(file)
    return mod


def create_vectorstore(docs):
    embeddings_model = HuggingFaceEmbeddings(model_name= "all-mpnet-base-v2")#, cache_folder = dir)
    vectorstore = None # incase there is an existing vectorstore (had problems with this but only when not in a function)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings_model)
    return vectorstore

    

class S3DirectoryLoader:
    """Alternative that uses s3fs instead of boto3 (struggle to install)
    """
    def __init__(self, bucket_name, prefix=""):
        self.bucket_name = bucket_name
        self.prefix = prefix

    def load(self):
        S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
        s3 = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})  # anon=False means it uses your AWS credentials

        try:
            # List all objects with the given prefix
            objects = s3.ls(f'{self.bucket_name}/{self.prefix}')
            data = {}

            for obj in objects:
                # Open and read each object
                with s3.open(obj, 'rb') as file:
                    file_content = file.read()
                    # Optionally process the file content based on its type
                    # For now, we'll just store the content in the dictionary
                    data[obj] = file_content

            return data

        except Exception as e:
            # Handle errors (e.g., authentication errors, file not found)
            print(f"An error occurred: {e}")
            return None



if __name__ == "__main__":
    # Set up target folder
    bucket = "sjentoft"
    prefix = f"db-files/dapla-manual/{dato}"

    docs = split_html(bucket, prefix)
    print("docs split")
    # print(docs[0])

    vectorstore = create_vectorstore(docs)
    print("vectore store done")
