import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf.type != 'application/pdf':
            st.error("Error: The uploaded file is not a PDF.")
            return ""
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except TypeError as e:
            st.error(f"Error reading PDF file: {e}")
            st.error("Please ensure that the PDF file is not corrupted and try again.")
            return ""
    return text

# Function to split text into chunks
def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to process URLs and return text chunks
def process_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(data)
    return docs

# Function to create a vector store from text chunks
def get_vectorstore_from_chunks(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = GooglePalm(google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0.1)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    prompt_template = """
    You are an AI assistant. Your responses should be detailed and informative, with a minimum of 300 words.
    
    User: {question}
    
    Assistant:
    """
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        prompt=prompt_template
    )
    return conversation_chain

# Function to handle user input and generate responses
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload PDFs or provide URLs before asking a question.")
        return
    
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
                    if raw_text:
                        text_chunks += get_text_chunks(raw_text)
                    else:
                        st.error("Failed to extract text from PDFs.")
                
                if any(url_inputs):
                    url_docs = process_urls([url for url in url_inputs if url])
                    if url_docs:
                        for doc in url_docs:
                            text_chunks += get_text_chunks(doc.page_content)
                    else:
                        st.error("Failed to fetch data from URLs.")
                
                if text_chunks:
                    vectorstore = get_vectorstore_from_chunks(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                else:
                    st.error("No valid PDFs or URLs provided or processed.")

if __name__ == '__main__':
    main()
