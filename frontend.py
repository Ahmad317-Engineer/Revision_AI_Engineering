import streamlit as st
import requests

st.title("🤖 RAG Document Assistant")
st.write("Ask questions about your documents")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                "http://localhost:8000/ask",
                json={"text": question}
            )
            data = response.json()
            st.write("**Answer:**")
            st.write(data["answer"])
        except Exception as e:
            st.error(f"Backend error: {e}")