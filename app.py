import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from langchain import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os
import sys
import os
from dotenv import load_dotenv

#load_dotenv()
openai_api_key = st.text_input("Enter your api key:", type="password")
#name = st.text_input("Enter your api key:", type="password")

# Your code here

document=[]
for file in os.listdir("docs"):
  if file.endswith(".pdf"):
    pdf_path="./docs/"+file
    loader=PyPDFLoader(pdf_path)
    document.extend(loader.load())
  elif file.endswith('.docx') or file.endswith('.doc'):
    doc_path="./docs/"+file
    loader=Docx2txtLoader(doc_path)
    document.extend(loader.load())
  elif file.endswith('.txt'):
    text_path="./docs/"+file
    loader=TextLoader(text_path)
    document.extend(loader.load())
#print(len(document))
document_splitter=CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
document_chunks=document_splitter.split_documents(document)
#print(len(document_chunks))




embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
vectordb=Chroma.from_documents(document_chunks,embedding=embeddings, persist_directory='./data')
vectordb.persist()
llm=ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo')
#print(llm)

memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#Create our Q/A Chain
pdf_qa=ConversationalRetrievalChain.from_llm(llm=llm,
                                             retriever=vectordb.as_retriever(search_kwargs={'k':6}),
                                             verbose=False, memory=memory)
#result=pdf_qa({"question":"tell me about sai aditya"})
#print(result['answer'])


user_question = st.text_input("Ask a Question from the PDF Files")

if user_question:  # Only execute pdf_qa if the user enters a question
    try:
        result = pdf_qa({"question": user_question})
        st.success("Here's the answer from the PDF files:")
        st.write(result['answer'])  # Display the result using st.write
    except Exception as e:
        st.error(f"An error occurred while processing your question: {e}")
st.success("Done")








