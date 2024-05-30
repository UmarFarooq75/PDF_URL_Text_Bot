## Introduction

The MultiPDF and URL Chat App is a Python application that allows you to chat with multiple PDF documents and web URLs. You can ask questions about the content using natural language, and the application will provide relevant responses based on the documents and web pages. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs and URLs.

## How It Works

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. **PDF and URL Loading:** The app reads multiple PDF documents and extracts their text content, as well as processes the text from provided URLs.
2. **Text Chunking:** The extracted text is divided into smaller chunks that can be processed effectively.
3. **Language Model:** The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
4. **Similarity Matching:** When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
5. **Response Generation:** The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs and URLs.

## Dependencies and Installation

To install the MultiPDF and URL Chat App, please follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables:**
   Obtain API keys from OpenAI and Google, and add them to the `.env` file in the project directory.
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

To use the MultiPDF and URL Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the API keys to the `.env` file.

2. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. **Load PDF Documents and Enter URLs:**
   - Upload multiple PDF documents into the app by following the instructions in the sidebar.
   - Enter the URLs you want to include for text processing.

5. **Ask Questions:**
   Use the chat interface to ask questions in natural language about the loaded PDFs and URLs.

## Contributing

This repository is intended for educational purposes and does not accept further contributions. It serves as supporting material for a YouTube tutorial that demonstrates how to build this project. Feel free to utilize and enhance the app based on your own requirements.

## License

The MultiPDF and URL Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).

---
