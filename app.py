import streamlit as st
from src.search import Search

st.set_page_config(page_title="Reddit GenAI Search Engine", layout="wide")
st.title("🔎 Reddit GenAI Search Engine")

@st.cache_resource
def load_search_engine():
    return Search(index_name="reddit-genai")

search_engine = load_search_engine()


query = st.text_input("Enter your search query:")
if query:
    with st.spinner("Searching..."):
        results = search_engine.search(query)
        # results["selftext_clean"] = results["selftext_clean"].fillna("").str.slice(0, 250) + "..."
        st.subheader("Answer from LLM:")
        st.write(results)
        # st.dataframe(
        #     results[["subreddit", "title", "selftext_clean", "created_day", "score"]].reset_index(drop=True),
        #     use_container_width=True,
        #     hide_index=True
        # )

        # with st.expander("See Raw Results Data"):
        #     st.json(results.head(5).to_dict(orient="records"))