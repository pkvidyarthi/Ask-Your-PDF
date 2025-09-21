# Importing Modules

from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from tiktoken._educational import *

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    # Display the header with a specific text color
    st.markdown(f'<h1 style="color: #D2042D;">Ask Your PDF</h1>', unsafe_allow_html=True)
    st.write("Discover answers within your uploaded PDF effortlessly.\nUpload your PDF and uncover answers tailored to its content with ease.")

    # Upload File
    pdf = st.file_uploader("Upload Your PDF", type = "pdf")

    # Extract The Text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into Chunks
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # Create Embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Show User Input
        user_question = st.text_input("Ask a query about your PDF : ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type = "stuff")
            response = chain.run(input_documents = docs, question = user_question)
            st.write(response)

if _name_ == "_main_":
    # st.run(main, debug=True)
    main()
