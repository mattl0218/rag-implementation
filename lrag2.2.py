# UI Implementation with RAG using Claude

import os
import sys
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
import constants

# Import API key
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
# os.environ["ANTHROPIC_API_KEY"] = constants.ANTHROPIC_API_KEY

###############
 # RAG setup #
###############

# Create the embedding model
embeddings = HuggingFaceEmbeddings()

# Define file paths
file_paths = [
    "/data/data.txt",
    "/data/data2.txt",
    "/data/data3.txt"
]

# Load content from multiple files
loaders = [TextLoader(file_path, encoding='utf-8') for file_path in file_paths]

# Create index
index = VectorstoreIndexCreator(
    embedding=embeddings,
    vectorstore_cls=FAISS
).from_loaders(loaders)

# Create ConversationalRetrievalChain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatAnthropic(model="claude-3-5-sonnet-20240620"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)



###############
 # Streamlit #
###############

# App title
st.title('RAG-powered Q&A System')

# Setup a session state for messages and chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template
prompt = st.chat_input('Ask a question')

# If the user hits enter
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Use the RAG system to get a response
    result = chain.invoke({"question": prompt, "chat_history": st.session_state.chat_history})
    response = result['answer']
    
    # Show the response
    st.chat_message('assistant').markdown(response)
    # Store the response in state
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    
    # Update chat history
    st.session_state.chat_history.append((prompt, response))
