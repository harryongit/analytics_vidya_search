# vector_store.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def prepare_documents(self, df):
        """Prepare course data for embedding"""
        documents = []
        for _, row in df.iterrows():
            text = (
                f"Title: {row['title']}\n"
                f"Description: {row['description']}\n"
                f"Level: {row['level']}\n"
                f"Duration: {row['duration']}"
            )
            chunks = self.text_splitter.split_text(text)
            documents.extend(chunks)
            
        logger.info(f"Prepared {len(documents)} text chunks for embedding")
        return documents
    
    def create_vector_store(self, documents, persist_directory="./chroma_db"):
        """Create and persist vector store"""
        try:
            vector_store = Chroma.from_texts(
                texts=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            vector_store.persist()
            logger.info(f"Vector store created and persisted to {persist_directory}")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
    
    def load_vector_store(self, persist_directory="./chroma_db"):
        """Load existing vector store"""
        try:
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            logger.info(f"Vector store loaded from {persist_directory}")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
    
    def run(self, csv_file='courses.csv'):
        """Run the complete vector store creation process"""
        df = pd.read_csv(csv_file)
        documents = self.prepare_documents(df)
        return self.create_vector_store(documents)

if __name__ == "__main__":
    vector_store_manager = VectorStoreManager()
    vector_store_manager.run()