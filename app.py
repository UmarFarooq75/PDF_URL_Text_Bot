import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from htmlTemplates import css, bot_template, user_template
from langchain.llms import GooglePalm

load_dotenv()  # Take environment variables from .env (especially openai API key)

# Define functions for processing PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Define functions for processing URLs
def process_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(data)
    return docs

# Create vector store from text chunks
def get_vectorstore_from_chunks(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Create conversation chain
def get_conversation_chain(vectorstore):
    llm = GooglePalm(google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0.1)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs and URLs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "url_fields" not in st.session_state:
        st.session_state.url_fields = 1

    st.header("Chat with PDFs and URLs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        # Dynamic URL input fields
        st.subheader("Enter URLs")
        url_inputs = [st.text_input(f"URL {i+1}", key=f"url_{i}") for i in range(st.session_state.url_fields)]
        
        add_url_button = st.button("Add another URL")
        if add_url_button:
            st.session_state.url_fields += 1
        
        process_data = st.button("Process Data")

        if process_data:
            text_chunks = []
            with st.spinner("Processing PDFs and URLs"):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks += get_text_chunks(raw_text)
                
                if any(url_inputs):
                    url_docs = process_urls([url for url in url_inputs if url])
                    for doc in url_docs:
                        text_chunks += get_text_chunks(doc.page_content)
                
                if text_chunks:
                    vectorstore = get_vectorstore_from_chunks(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    st.error("No valid PDFs or URLs provided.")

if __name__ == '__main__':
    main()
