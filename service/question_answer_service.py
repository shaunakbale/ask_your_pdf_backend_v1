import os
import tempfile

from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

#Hello World
load_dotenv()

class QuestionAnswerService:
    # RAG Prompt Template
    prompt = hub.pull("rlm/rag-prompt")

    def __init__(self):
        self.retriever = None

    # Load PDF and Indexing
    def load_pdf(self, pdf_bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_bytes)
            temp_file_path = temp_file.name  # Get the path of the temporary file

        loaders = [PyPDFLoader(temp_file_path)]

        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        ).from_loaders(loaders)

        self.retriever = index.vectorstore.as_retriever()  # Set the retriever after loading the PDF
        return index

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_response(self, question):
        if self.retriever is None:
            raise ValueError("PDF not loaded. Please load a PDF file before generating a response.")

        # Create the RAG chain
        rag_chain = (
                {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
                | self.prompt
                | ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.getenv("GEMINI_API"))  # Replace with your API key
                | StrOutputParser()
        )

        # Invoke the chain with the provided question
        response = rag_chain.invoke(question)
        return response

    def clear_pdf(self):
        """Clear the currently loaded PDF and retriever"""
        self.retriever = None
