# UCM GPT: University Chatbot Project

UCM GPT is a personal assistant designed specifically for students at the University of Central Missouri. It helps students with course information, campus services, and general inquiries by providing quick and accurate assistance.

## Features

- **Instant Answers:** Get immediate answers to frequently asked questions about courses, programs, and admission.
- **Campus Information:** Find information about campus facilities, such as libraries, labs, and classroom options.

## Project Team

- **Sai Aditya Guntupalli - 700757316**
- **Mohith Degala - 700746278**
- **Murali Krishna Ponnam - 700755557**
- **Keerthy Pabbathineni - 700747373**

### Instructor

- **Muhammad Zubair Khan**

## Requirements

- `streamlit`
- `langchain`
- `pysqlite3`
- `chromadb`
- `torch`
- `transformers`
- `dotenv`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/ucm-gpt.git
    cd ucm-gpt
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Add your OpenAI API key to `.streamlit/secrets.toml`:

    ```toml
    [secrets]
    openai_api_key = "your_openai_api_key"
    ```

## Usage

1. Place your documents in the `docs` directory. Supported file formats are PDF, DOCX, and TXT.

2. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

3. Open the web application in your browser. You will see the UCM GPT interface where you can ask questions and get responses based on the documents you have uploaded.

## Code Overview

### `app.py`

- Imports necessary libraries and modules.
- Displays the title and introductory text using Streamlit.
- Loads documents from the `docs` directory.
- Splits documents into chunks using `CharacterTextSplitter`.
- Embeds document chunks using OpenAI embeddings.
- Creates a Chroma vector store from document chunks.
- Initializes a Conversational Retrieval Chain for question answering.
- Provides a text input for user questions and displays the responses.
