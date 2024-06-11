import urllib.parse
import logging
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import base64
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from io import BytesIO

load_dotenv()

# Set the logging level to INFO for HTTPX library
logging.getLogger("httpx").setLevel(logging.INFO)
# Set the logging level to INFO for pymongo
logging.getLogger("pymongo").setLevel(logging.INFO)


username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
db_name=os.getenv('DB_NAME')
encoded_username = urllib.parse.quote_plus(username)
encoded_password = urllib.parse.quote_plus(password)

# MongoDB Atlas connection details
mongo_uri = f"mongodb+srv://{encoded_username}:{encoded_password}@cluster0.w7vw3w2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Initialize the MongoDB client
client = MongoClient(mongo_uri)
db = client[db_name]
print("MongoDB initialized successfully!")
collection_names = db.list_collection_names()

# Print the collection names
print("Collections in the database:")
for collection_name in collection_names:
    print(collection_name)
google_api_key=os.getenv("GOOGLE_API_KEY2")
COH_API_KEY=os.getenv('COHERE_API_KEY')
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-1.5-pro")
llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=google_api_key)

class BinaryDataLoader(BaseLoader):
    def __init__(self, binary_data: bytes, file_name: str):
        self.binary_data = binary_data
        self.file_name = file_name

    def load(self) -> Document:
        # Encode binary data to base64 string
        encoded_data = base64.b64encode(self.binary_data).decode('utf-8')
        
        # Create a Document object
        document = Document(
            page_content=encoded_data,
            metadata={"source": self.file_name}
        )
        
        return document
    
def parse_query(query):
    # Example query: "Tell me about Quantum Mechanics from semester 3"
    match = re.search(r'about (.+?) from semester (\d+)', query, re.IGNORECASE)
    if not match:
        return None, None
    
    subject = match.group(1).strip()
    semester = match.group(2).strip()
    return semester, subject

def generate_mongo_query(semester, subject):
    collection_name = f"semester_{semester}"
    mongo_query = {
        "file_name": {"$regex": subject, "$options": "i"}
    }
    return collection_name, mongo_query

def process_query(query):
    # Parse the query to get semester and subject
    semester, subject = parse_query(query)
    if not semester or not subject:
        return "Could not parse the query. Please ensure it contains both a semester number and a subject name."
    
    # Generate MongoDB query
    collection_name, mongo_query = generate_mongo_query(semester, subject)
    
    # Execute the MongoDB query
    collection = db[collection_name]
    results = list(collection.find(mongo_query, {"file_name": 1, "file_data": 1}))
    
    if not results:
        return f"No documents found for subject '{subject}' in semester {semester}."
    
    # Extract and load information from the documents
    relevant_info = []
    for result in results:
        file_name = result.get('file_name', 'Unknown')
        file_data = result.get('file_data', b'')

        # Use BinaryDataLoader to process binary data
        loader = BinaryDataLoader(file_data, file_name)
        document = loader.load()
        
        # Debug print statement
        print(f"Loaded document from {file_name} with content length {len(document.page_content)} bytes")

        relevant_info.append(document)
    
    return relevant_info


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        pdf_stream = BytesIO(base64.b64decode(pdf_doc.page_content))
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=COH_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro',google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0",cohere_api_key=COH_API_KEY)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    response = chain.invoke({"context": context, "question": user_question}, return_only_outputs=True)
    print("\nResponse:", response)
    
    # Adjust the key according to the actual response structure
    return response.get("text", "No output text found")


def query_and_ask(query, user_question):
    relevant_info = process_query(query)
    if isinstance(relevant_info, list):
        pdf_text = get_pdf_text(relevant_info)
        text_chunks = get_text_chunks(pdf_text)
        get_vector_store(text_chunks)
        response = user_input(user_question)
        st.write("\nReply: ", response)
    else:
        st.write(relevant_info)

def main():
    st.set_page_config(page_title="Semester Scholar", page_icon="📚", layout="wide")

    st.title("📚 Semester Scholar")
    st.markdown("### Retrieve and query your semester materials effortlessly!")

    query_placeholder = "e.g., Tell me about Data and Information Security from Semester 6"
    user_question_placeholder = "e.g., What are the components of an information system?"

    query = st.text_input("Enter document name and semester number",placeholder=query_placeholder)
    user_question = st.text_input("Enter your question",placeholder=user_question_placeholder)

    if st.button("Process Query"):
        with st.spinner("Thinking..."):
            query_and_ask(query,user_question)

    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. **Enter Query**: Type in the subject and semester (e.g., "Tell me about Cloud Computing from semester 6").

                          
    2. **Ask Question**: Type in your specific question about the subject (e.g., "What are the various cloud models?").

                         
    3. **Process Query**: Click the button to process and retrieve the information.
                        

    4. **Get Answer**: Receive the detailed answer to your question.
    """)

if __name__ == "__main__":
    main()
