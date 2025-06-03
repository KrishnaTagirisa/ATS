# -------------------------------
#  Import Required Libraries
# -------------------------------
import os  # For file and directory operations
import streamlit as st  # Streamlit for creating web UI
from PyPDF2 import PdfReader  # To read and extract text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split large text into manageable chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # To convert text to embeddings using Google AI
import google.generativeai as genai  # Google's Generative AI SDK for configuration
from langchain_community.vectorstores import FAISS  # FAISS: for storing and querying text embeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # LangChain wrapper for Google's chat model (Gemini)
from langchain.chains.question_answering import load_qa_chain  # QA chain for handling question answering
from langchain.prompts import PromptTemplate  # To create a custom prompt template
from dotenv import load_dotenv  # To load environment variables from a .env file

# -------------------------------
#  Load Environment Variables
# -------------------------------
load_dotenv()  # Loads the .env file variables into environment
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Configure Google Generative AI using API key

# Optional for deployment on Streamlit Cloud (hardcoded API key can be used)
# genai.configure(api_key="XXXXX")  

# -------------------------------
#  Setup Directory for PDF Uploads
# -------------------------------
PDF_DIR = "uploaded_pdfs"  # Directory to store uploaded PDFs
os.makedirs(PDF_DIR, exist_ok=True)  # Create directory if it doesn't exist

# -------------------------------
#  Save Uploaded PDFs
# -------------------------------
def save_uploaded_files(uploaded_files):
    """Save uploaded PDF files to the local directory."""
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Write the file contents
        saved_paths.append(file_path)
    return saved_paths

# -------------------------------
#  Extract Text from PDFs
# -------------------------------
def get_pdf_text(pdf_paths):
    """Extracts all text from the provided list of PDF files."""
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text  # Concatenate page text
            except Exception as e:
                print(f"Error extracting text from page: {e}")
    return text

# -------------------------------
# List Uploaded PDF Files
# -------------------------------
def list_stored_pdfs():
    """Lists all PDF files stored in the uploaded_pdfs folder."""
    return [f for f in os.listdir(PDF_DIR) if os.path.isfile(os.path.join(PDF_DIR, f))]

# -------------------------------
#  Split Large Text into Chunks
# -------------------------------
def get_text_chunks(text):
    """Splits large text into smaller overlapping chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# -------------------------------
#  Create Vector Store with Embeddings
# -------------------------------
def get_vector_store(text_chunks):
    """Creates FAISS vector store from text chunks using Google embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save vector index locally for later use

# -------------------------------
#  Create Conversational QA Chain
# -------------------------------
def get_conversational_chain():
    """Initializes a question-answering chain using Google's Gemini chat model."""
    prompt_template = """
    Act as an AI-PDF expert. Users upload one or more PDF files and ask you questions based on those files.
    Your job is to understand the question and generate detailed answers based on the PDF content.
    Identify relevant paragraphs and combine them into a complete, accurate answer.
    If the answer isn't in the context, respond with "answer is not available in the context".
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# -------------------------------
#  Handle User Question
# -------------------------------
def user_input(user_question):
    """Processes user question and generates a response based on uploaded PDFs."""
    detailed_question = user_question + " Explain in detail."
    
    # Load embeddings and existing FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform semantic search to find relevant chunks
    docs = new_db.similarity_search(detailed_question)
    
    # Generate answer using conversational chain
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": detailed_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        if "stop condition" in str(e):
            st.write("Reply:", "Sorry, I couldn't find an answer based on the provided context.")
        else:
            raise e  # Re-raise other unexpected exceptions

# -------------------------------
#  Main Streamlit App Logic
# -------------------------------
def main():
    """Main function to run the Streamlit app interface."""
    st.set_page_config("Chat PDF")

    # ---------------------------
    #  Dark Theme Styling
    # ---------------------------
    st.markdown(
        """
        <style>
        body {
            background-color: black;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #333;
            color: white;
        }
        .stButton>button {
            border: 2px solid #4CAF50;
            background-color: #333;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------
    #  App Title
    # ---------------------------
    st.title("Chat with Your PDFs")

    # ---------------------------
    # Sidebar: Upload PDFs
    # ---------------------------
    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader("Upload PDF files here", type="pdf", accept_multiple_files=True)
        if st.button("Process Uploaded PDFs"):
            with st.spinner("Processing..."):
                pdf_paths = save_uploaded_files(uploaded_files)  # Save files
                raw_text = get_pdf_text(pdf_paths)  # Extract text
                text_chunks = get_text_chunks(raw_text)  # Split into chunks
                get_vector_store(text_chunks)  # Generate embeddings & store in FAISS
                st.success("PDFs processed successfully!")

        #  Show list of stored PDFs
        st.header("Stored PDFs")
        for pdf_file in list_stored_pdfs():
            st.text(pdf_file)

    # ---------------------------
    #  Main Panel: Ask Questions
    # ---------------------------
    st.header("Ask a Question")
    user_question = st.text_input("Enter your question here:")
    if user_question:
        user_input(user_question)  # Generate answer based on uploaded PDF content

# -------------------------------
#  Run App
# -------------------------------
if __name__ == "__main__":
    main()
