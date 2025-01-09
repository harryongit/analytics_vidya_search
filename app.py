# app.py
from app.core.scraper import CourseScraper
from app.core.vector_store import VectorStoreManager
from app.core.search_system import CourseSearchSystem
import streamlit as st
import os

def initialize_system():
    """Initialize or load the search system"""
    vector_store_manager = VectorStoreManager()
    
    # Check if we need to create new vector store
    if not os.path.exists("./chroma_db"):
        st.info("First-time setup: Collecting course data...")
        scraper = CourseScraper()
        df = scraper.run()
        
        st.info("Creating search index...")
        vector_store = vector_store_manager.run()
    else:
        vector_store = vector_store_manager.load_vector_store()
    
    return CourseSearchSystem(vector_store)

def main():
    st.title("Analytics Vidhya Course Search")
    st.write("Find the perfect free course using natural language search!")
    
    # Initialize search system
    search_system = initialize_system()
    
    # Create search interface
    query = st.text_input(
        "What would you like to learn?",
        placeholder="e.g., 'Python courses for beginners' or 'Advanced machine learning courses'"
    )
    
    if query:
        with st.spinner("Searching for relevant courses..."):
            results = search_system.search(query)
            
            st.write("### Recommended Courses")
            st.write(results)
    
    # Add helpful tips
    with st.expander("Search Tips"):
        st.write("""
        - Be specific about your skill level (beginner, intermediate, advanced)
        - Mention specific topics you're interested in
        - Include any time constraints (e.g., "short courses" or "comprehensive courses")
        - Specify your learning goals
        """)

if __name__ == "__main__":
    main()
