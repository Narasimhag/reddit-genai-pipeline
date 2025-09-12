import streamlit as st
from src.search import Search

st.set_page_config(page_title="Reddit GenAI Search Engine", layout="wide")
st.title("🔎 Reddit GenAI Search Engine")

@st.cache_resource
def load_search_engine():
    return Search(index_name="reddit-genai", top_k_retrieve=150, top_k_rerank=20)

search_engine = load_search_engine()


query = st.text_input("Enter your search query:")
if query:
    with st.spinner("Searching..."):
        results = search_engine.search(query)
        st.subheader("Answer from LLM:")
        st.write(results)
