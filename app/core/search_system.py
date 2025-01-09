# search_system.py
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseSearchSystem:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingface_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )
        self.qa_chain = self._setup_retrieval_chain()
        
    def _setup_retrieval_chain(self):
        """Set up the retrieval chain"""
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever
            )
            logger.info("Retrieval chain setup completed")
            return qa_chain
        except Exception as e:
            logger.error(f"Error setting up retrieval chain: {e}")
            return None
    
    def search(self, query):
        """Process search query and return results"""
        try:
            response = self.qa_chain.run(query)
            logger.info("Search query processed successfully")
            return response
        except Exception as e:
            logger.error(f"Error processing search query: {e}")
            return "Sorry, there was an error processing your search query."