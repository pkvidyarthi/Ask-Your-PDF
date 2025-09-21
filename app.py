from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from openai import RateLimitError  # ✅ Correct import

# Load environment variables
load_dotenv()

def main():
    st.set_page_config(page_title="Ask Your PDF")

    # Header
    st.markdown('<h1 style="color: #D2042D;">Ask Your PDF</h1>', unsafe_allow_html=True)
    st.write("Discover answers within your uploaded PDF effortlessly.\nUpload your PDF and uncover answers tailored to its content with ease.")

    # Upload File
    pdf = st.file_uploader("Upload Your PDF", type="pdf")

    if pdf is not None:
        # Extract text
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=3000,
            chunk_overlap=300,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # 🚨 Limit chunks for free quota
        chunks = chunks[:10]

        try:
            # Create embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # User Input
            user_question = st.text_input("Ask a query about your PDF : ")
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)

        except RateLimitError:
            st.error("🚨 OpenAI API rate limit exceeded. Please try again later or check your credits.")


if __name__ == "__main__":
    main()
