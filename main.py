import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CourseAssistant")

# Suppress noisy library warnings for a cleaner CLI experience
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# Path Configuration (Must match ingest.py)
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"
BM25_PATH = DB_DIR / "bm25_retriever.pkl"

# Model Configuration
COLLECTION_NAME = "course_materials"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"  # SOTA Open Source Model

class CourseAssistant:
    """
    A RAG-based assistant that answers student questions using a hybrid search strategy
    (Semantic Search + Keyword Search) over course materials.
    """

    def __init__(self):
        self._validate_environment()
        
        logger.info("Initializing Course Assistant...")
        
        # 1. Initialize Embeddings
        logger.info(f"Loading Embeddings: {EMBEDDING_MODEL_NAME}")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # 2. Load Vector Store (Chroma)
        logger.info(f"Loading Vector Database from: {DB_DIR}")
        self.vector_store = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        self.chroma_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # 3. Load Sparse Retriever (BM25)
        self.bm25_retriever = self._load_bm25()

        # 4. Create Hybrid Ensemble Retriever
        # Weighted 50/50: Balances exact keyword matches (BM25) with conceptual matches (Chroma)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=[0.4, 0.6]
        )

        # 5. Initialize LLM
        self.llm = ChatGroq(
            temperature=0.0,  # Deterministic for academic accuracy
            model_name=LLM_MODEL_NAME,
            api_key=os.getenv("GROQ_API_KEY")
        )

        # 6. Build the RAG Chain
        self.chain = self._build_rag_chain()
        logger.info("Assistant is ready to serve.")

    def _validate_environment(self):
        """Ensures all necessary files and keys are present before starting."""
        if not os.getenv("GROQ_API_KEY"):
            logger.critical("GROQ_API_KEY is missing. Please check your .env file.")
            sys.exit(1)
            
        if not DB_DIR.exists() or not BM25_PATH.exists():
            logger.critical("Database not found. Please run 'python ingest.py' first.")
            sys.exit(1)

    def _load_bm25(self):
        """Loads the pre-computed BM25 index from disk."""
        try:
            with open(BM25_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            sys.exit(1)

    def _build_rag_chain(self):
        """Constructs the LangChain pipeline."""
        
        # System Prompt: Enforces strict adherence to context
        template = """
        You are a helpful Course Assistant. 
        Answer the student's question based on the Context provided below.
        
        INSTRUCTIONS:
        1. Base your answer ONLY on the Context.
        2. If the Context contains the answer, write a clear explanation and cite the source file (e.g., [Source: lecture1.pptx]).
        3. If the Context does NOT contain the answer, simply state: "I cannot answer this question as it is not covered in the course materials."
        4. Do not apologize. Be direct and academic.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        return (
            {"context": self.ensemble_retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a clean string for the LLM."""
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.replace("\n", " ")
            formatted.append(f"[Source: {source}]\nContent: {content}\n")
        return "\n---\n".join(formatted)

    def get_answer(self, query: str) -> str:
        """Public method to get an answer for a query."""
        try:
            return self.chain.invoke(query)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "An error occurred while processing your request."

# --- Main Execution Loop ---
def main():
    print("\n" + "="*60)
    print("NLP COURSE ASSISTANT (RAG SYSTEM)")
    print("="*60)
    print("Loading system... please wait.\n")

    try:
        bot = CourseAssistant()
        print("\n" + "-"*60)
        print("Ready! Ask a question about your course materials.")
        print("   Type 'quit', 'exit', or 'q' to stop.")
        print("-"*60 + "\n")

        while True:
            user_input = input("Student: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye! Good luck with your studies.")
                break
            
            if not user_input:
                continue

            print("Assistant: Thinking...", end="\r")
            response = bot.get_answer(user_input)
            
            # Clear "Thinking..." line and print response
            print(" " * 30, end="\r") 
            print(f"Assistant: {response}\n")
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n\nSystem shutting down.")
    except Exception as e:
        logger.critical(f"Fatal Error: {e}")

if __name__ == "__main__":
    main()