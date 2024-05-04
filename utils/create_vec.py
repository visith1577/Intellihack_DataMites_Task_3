import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

loader = PyPDFLoader("data/data.pdf")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embedding_function = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embedding_function)

query = "Eligibility Criteria for housing loans"

# # save to disk
# db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
# docs = db2.similarity_search(query)

db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search(query)
print(docs[0].page_content)
