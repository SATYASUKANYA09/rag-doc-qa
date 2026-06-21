import streamlit as st
import tempfile
import os
from rag_engine import process_pdf, ask_question

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")

st.title("📄 AI Document Q&A")
st.write("Upload a PDF and ask questions about it!")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing your PDF..."):
        st.session_state.vectorstore = process_pdf(tmp_path)
    
    os.unlink(tmp_path)
    st.success("✅ PDF processed! Ask your questions below.")

if st.session_state.vectorstore is not None:
    question = st.text_input("Ask a question about your document:")
    
    if question:
        with st.spinner("Thinking..."):
            answer = ask_question(st.session_state.vectorstore, question)
        st.markdown("### Answer:")
        st.write(answer)