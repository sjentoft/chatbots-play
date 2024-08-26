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
from langchain.text_splitter import CharacterTextSplitter

from google.cloud import storage #poetry add google-cloud-storage
from langchain_core.document_loaders import BaseLoader
from langchain.schema import Document

from bs4 import BeautifulSoup

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

#from .auth import AuthClient
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-file.json"
#import boto3

#print(os.environ["AWS_ACCESS_KEY_ID"])
#print(os.environ["AWS_SECRET_ACCESS_KEY"])
#print(os.getenv('AWS_SESSION_TOKEN')) 


dato = "2024-08-16"


def split_html(bucket, prefix):
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    txt_loader = S3DirectoryLoader(bucket, prefix)
        # glob="**/*.html", 
        #endpoint_url = S3_ENDPOINT_URL )
    
    #txt_loader.load()
    #print(txt_loader)
    #loaders = [txt_loader]
    documents = []

    try:
        raw_documents = txt_loader.load()
        # Extract the content from the loaded documents
        for key, content in raw_documents.items():
            text_content = content.decode('utf-8') if isinstance(content, bytes) else content

            # Parse HTML and extract text
            soup = BeautifulSoup(text_content, 'html.parser')
            clean_text = soup.get_text(separator='\n', strip=True)
            
            # Wrap the cleaned text in a Document object
            documents.append(Document(page_content=clean_text))

    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return []

    # Initialize the text splitter
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    
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
    #target_folder = f"{bucket}/{folder}/{dato}"

    # Set up fs connection
    #S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    #fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})

    # 
    docs = split_html(bucket, prefix)
    print("docs split")
    # print(docs[0])
    # Test created document loader
    #dl = S3DirectoryLoader(bucket, f'{folder}/{dato}')
    #docs = dl.load()
    #print(docs[0])
    # Vectore Store
    #embed_model = get_embedding_model(configs["embedding_model"])
    vectorstore = create_vectorstore(docs)
    print("vectore store done")
    #embeddings = MyEmbeddings()
    #splitter = SemanticChunker(embeddings)


""" Old below
from google.cloud import storage
from langchain_core.document_loaders import BaseLoader
from langchain.schema import Document

#fs = dp.FileClient()

#from google.cloud import storage

#bucket = "gs://ssb-play-chatbot-data-produkt-prod"
#folder = "db-files/docs"

fs = dp.FileClient()
fs.get_versions(bucket, folder)

# Usage example
bucket = "gs://ssb-play-chatbot-data-produkt-prod"
folder = f"db-files/docs"

loader = GCSDirectoryLoader(bucket_name=bucket, prefix=folder)
documents = loader.load()

for doc in documents:
    print(doc.metadata)
    print(doc.text)

"""