# # Create Chroma DB for dapla chatbot

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from bs4 import BeautifulSoup
import re

from config import dato, embeddings_model, path_to_manual, path_to_chromadb


def split_html(path, chunk_size=512, chunk_overlap = 0):
    txt_loader = DirectoryLoader(path, glob="**/*.html")
    documents = []

    try:
        raw_documents = txt_loader.load()

        #for key, content in doc_items:
        for content in raw_documents:

            text_content = content.decode('utf-8') if isinstance(content, bytes) else content
            clean_text = text_content.page_content

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
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Split the documents into chunks
    chunks = text_splitter.split_documents(documents)
    
    return chunks


def remove_single_word_lines(text):
    # Split text into lines
    lines = text.split('\n')
    
    # Filter out lines that contain only a single word
    filtered_lines = [line for line in lines if len(re.findall(r'\w+', line)) > 1]
    
    # Join the remaining lines back into a single string
    return '\n'.join(filtered_lines)


def create_vectorstore(docs, model):
    embeddings_model = HuggingFaceEmbeddings(model_name=model)
    vectorstore = None # incase there is an existing vectorstore (had problems with this but only when not in a function)
    vectorstore = Chroma.from_documents(docs, embedding=embeddings_model,
                                        persist_directory = path_to_chromadb)
    return vectorstore
    

if __name__ == "__main__":
    # Folder to files
    folder = f"{path_to_manual}/{dato}/statistikkere"
    print(f'Collecting files for chunking from: {folder}')

    # Split and clean documents
    docs = split_html(path = folder)
    print("Docs split")

    # Vectore Store
    vectorstore = create_vectorstore(docs, model = embeddings_model)
    print("Vectore store done")
    