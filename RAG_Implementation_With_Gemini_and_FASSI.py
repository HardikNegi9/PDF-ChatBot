import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf = PdfReader(pdf_doc)
        for page in pdf.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)




def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store= FAISS.from_texts(text_chunks,embeddings)
    vector_store.save_local("vector_store")




def get_qa_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided **context**, making sure to provide all the necessary information. If the answer is not in the provided context, just say "I don't know", don't make up information.
    
    Context: {context}
    Question: {question}
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    return chain




def user_input(user_ques):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    docs = new_vector_store.similarity_search(user_ques)
    chain = get_qa_chain()

    response = chain.invoke({
        "context": docs, "question": user_ques})
    print(response)
    st.write(response)

def main():
    st.set_page_config(page_title="Document QA", page_icon="📚")
    st.header("Document QA")

    user_ques = st.text_input("Enter your question here")
    
    if user_ques:
        user_input(user_ques)
    
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(text)
                get_vector_store(text_chunks)
                st.write("PDF processed successfully")

if __name__ == "__main__":
    main()
