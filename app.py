import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Offline PDF RAG Assistant",
    page_icon="📄"
)


# ==========================================
# TITLE
# ==========================================

st.title("📄 Offline PDF RAG Assistant")

st.write(
    "Ask questions about your PDF using a fully local AI model."
)


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    pdf_path = "data/uploaded.pdf"

    with open(pdf_path, "wb") as file:
        file.write(uploaded_file.getbuffer())


    # --------------------------------------
    # LOAD PDF
    # --------------------------------------

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()


    # --------------------------------------
    # SPLIT TEXT
    # --------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)


    # --------------------------------------
    # EMBEDDINGS
    # --------------------------------------

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )


    # --------------------------------------
    # CHROMADB
    # --------------------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_rag"
    )


    # --------------------------------------
    # RETRIEVER
    # --------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


    # --------------------------------------
    # OLLAMA MODEL
    # --------------------------------------

    model = ChatOllama(
        model="phi3:mini",
        temperature=0
    )


    # --------------------------------------
    # PROMPT
    # --------------------------------------

    prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF assistant.

Answer the user's question using ONLY the context
provided below.

If the answer is not present in the context, say:

"I could not find the answer in the PDF."

Context:
{context}

Question:
{question}

Give a clear and concise answer.
""")


    # --------------------------------------
    # OUTPUT PARSER
    # --------------------------------------

    parser = StrOutputParser()


    st.success("PDF processed successfully! ✅")


    # ======================================
    # QUESTION INPUT
    # ======================================

    question = st.text_input(
        "Ask a question about your PDF:"
    )


    # ======================================
    # ASK BUTTON
    # ======================================

    if st.button("Ask Question"):

        if question.strip() == "":
            st.warning("Please enter a question.")

        else:

            # -------------------------------
            # RETRIEVE RELEVANT CHUNKS
            # -------------------------------

            relevant_documents = retriever.invoke(
                question
            )


            # -------------------------------
            # CREATE CONTEXT
            # -------------------------------

            context = "\n\n".join(
                document.page_content
                for document in relevant_documents
            )


            # -------------------------------
            # CREATE PROMPT
            # -------------------------------

            final_prompt = prompt.invoke({
                "context": context,
                "question": question
            })


            # -------------------------------
            # ASK PHI-3
            # -------------------------------

            with st.spinner("Thinking..."):

                response = model.invoke(
                    final_prompt
                )


            # -------------------------------
            # PARSE RESPONSE
            # -------------------------------

            answer = parser.invoke(response)


            # =================================
            # DISPLAY ANSWER
            # =================================

            st.subheader("🤖 Answer")

            st.write(answer)


            # =================================
            # SOURCES
            # =================================

            st.subheader("📚 Sources")

            pages = sorted(
                set(
                    document.metadata.get("page", 0) + 1
                    for document in relevant_documents
                )
            )

            for page in pages:
                st.write(f"Page {page}")