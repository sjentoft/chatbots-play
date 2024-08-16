# # Create Chroma DB for dapla chatbot

# To restore poetry env:
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


def split_html(dir):
    txt_loader = DirectoryLoader(f"./{dir}", glob="**/*.html")
    
    loaders = [txt_loader]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size = 512, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def dpDirectoryLoader(dir, file_type = "**/*.html"):
    txt_loader = S3DirectoryLoader(f"{dir}", glob="**/*.html")


def get_embedding_model(model_name):
    file_name = f'./local_model/{model_name}.pkl'
    with open(file_name, 'rb') as file:
        mod = pickle.load(file)
    return mod


def create_vectorstore(docs, embeddings_model):
    vectorstore = None # incase there is an existing vectorstore (had problems with this but only when not in a function)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings_model)
    return vectorstore


class GCSDirectoryLoader(BaseLoader):
    def __init__(self, bucket_name, prefix=""):
        self.bucket_name = bucket_name
        self.prefix = prefix

    def load(self):
        bucket = storage.Client().get_bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=self.prefix)
        
        documents = []
        for blob in blobs:
            content = blob.download_as_text()
            document = Document(
                text=content,
                metadata={"bucket": self.bucket_name, "path": blob.name}
            )
            documents.append(document)
        return documents


if __name__ == "__main__":
    # Set up target folder
    bucket = "sjentoft"
    folder = "db-files/dapla-manual"
    dato = date.today()
    target_folder = f"{bucket}/{folder}/{dato}"

    # Set up fs connection
    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})


    dl = GCSDirectoryLoader(bucket, f'{folder}/{dato}')
    docs = dl.load()
    #embed_model = get_embedding_model(configs["embedding_model"])
    #vectorstore = create_vectorstore(docs, embed_model)

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