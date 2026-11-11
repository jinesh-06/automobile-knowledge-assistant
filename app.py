import streamlit as st
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set page config FIRST
st.set_page_config(
    page_title="AI Automobile Knowledge Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

from modules.chroma_store import ChromaStore
from modules.embeddings import EmbeddingManager
from modules.document_manager import DocumentManager
from modules.retriever import Retriever
from modules.rag_pipeline import RAGPipeline

# Initialize backend components (cached to avoid reloading)
@st.cache_resource
def init_system():
    embedding_manager = EmbeddingManager()
    chroma_store = ChromaStore()
    doc_manager = DocumentManager(chroma_store, embedding_manager)
    retriever = Retriever(chroma_store, embedding_manager)
    return embedding_manager, chroma_store, doc_manager, retriever

embedding_manager, chroma_store, doc_manager, retriever = init_system()

# Ensure session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "gemini"
if "top_k" not in st.session_state:
    st.session_state.top_k = 6
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.2

# Sidebar Navigation
st.sidebar.title("🚗 Auto Knowledge AI")
nav_option = st.sidebar.radio(
    "Navigation", 
    ["🏠 Dashboard", "📄 Documents", "🔎 Search", "💬 Ask Assistant", "⚖ Compare", "📚 Knowledge Base", "⚙ Settings"]
)

# ----------------- 🏠 DASHBOARD -----------------
if nav_option == "🏠 Dashboard":
    st.title("🏠 Dashboard")
    stats = chroma_store.get_collection_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", stats["documents"])
    col2.metric("Pages", stats["pages"])
    col3.metric("Chunks", stats["chunks"])
    col4.metric("Vectors", stats["vectors"])
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.info(f"**LLM Provider:**\n\n{st.session_state.llm_provider.capitalize()}")
    col2.success("**Vector Database:**\n\nChromaDB (Active)")
    col3.warning(f"**Knowledge Base:**\n\n{stats['documents']} Documents")
    
# ----------------- 📄 DOCUMENTS -----------------
elif nav_option == "📄 Documents":
    st.title("📄 Document Management")
    
    uploaded_files = st.file_uploader("Upload Automobile PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_bytes = uploaded_file.read()
                
                with st.spinner(f"Processing {file_name}..."):
                    try:
                        # Process and check replace logic is inside but we can handle UI part here
                        result = doc_manager.process_and_index_document(file_bytes, file_name, replace_existing=True)
                        if result.get("warnings"):
                            for warn in result["warnings"]:
                                st.warning(warn)

                        if result.get("status") == "partial_success":
                            st.warning(f"""
                            **{file_name}** (Partial Extraction)
                            ✓ Uploaded
                            ⚠️ Only {result['pages']} of {result.get('total_pages', result['pages'])} pages extracted
                            ✓ {result['chunks']} chunks created
                            ✓ Embeddings generated
                            ✓ Added to ChromaDB
                            """)
                        elif result.get("status") == "partial_failure":
                            st.error(f"""
                            **{file_name}** (Partial Failure)
                            ✓ Physical file updated
                            ✓ New chunks inserted ({result['chunks']} chunks)
                            ❌ Stale chunk cleanup failed: {result.get('error', 'Database error')}
                            💡 Recovery: {result.get('recovery_message', 'Retry uploading the document with replacement enabled.')}
                            """)
                        else:
                            st.success(f"""
                            **{file_name}**
                            ✓ Uploaded
                            ✓ Text extracted & OCR checked
                            ✓ {result['chunks']} chunks created
                            ✓ Embeddings generated
                            ✓ Added to ChromaDB
                            """)
                    except Exception as e:
                        st.error(f"Error processing {file_name}: {str(e)}")
                        
    st.markdown("---")
    st.subheader("Indexed Documents")
    docs = chroma_store.get_all_documents_metadata()
    if not docs:
        st.info("No documents uploaded yet.")
    else:
        for doc in docs:
            col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
            col1.write(f"📄 **{doc['name']}**")
            col2.write(f"Pages: {doc['pages']}")
            col3.write(f"Chunks: {doc['chunks']}")
            if col4.button("Delete", key=f"del_{doc['name']}"):
                doc_manager.delete_document(doc['name'])
                st.rerun()

# ----------------- 🔎 SEARCH -----------------
elif nav_option == "🔎 Search":
    st.title("🔎 Semantic Search")
    
    query = st.text_input("Enter search query (e.g., 'engine overheating')")
    
    docs = chroma_store.get_all_documents_metadata()
    doc_options = ["All Documents"] + [d["name"] for d in docs]
    filter_doc = st.selectbox("Search in:", doc_options)
    
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            results = retriever.retrieve(query, top_k=st.session_state.top_k, filter_document_name=filter_doc, threshold=0.1)
            
            if not results:
                st.warning("No relevant information found.")
            else:
                for idx, res in enumerate(results):
                    with st.expander(f"Result {idx+1}: {res['metadata']['document_name']} (Page {res['metadata']['page_number']}) - Relevance: {res['relevance']}"):
                        st.write(res['text'])

# ----------------- 💬 ASK ASSISTANT -----------------
elif nav_option == "💬 Ask Assistant":
    st.title("💬 Ask Assistant")
    
    docs = chroma_store.get_all_documents_metadata()
    doc_options = ["All Documents"] + [d["name"] for d in docs]
    filter_doc = st.selectbox("Search in:", doc_options, key="chat_filter")
    
    st.markdown("**Example Questions:**")
    cols = st.columns(3)
    if cols[0].button("What type of engine oil is recommended?"):
        st.session_state.example_q = "What type of engine oil is recommended?"
    if cols[1].button("What is the recommended tyre pressure?"):
        st.session_state.example_q = "What is the recommended tyre pressure?"
    if cols[2].button("What should I do if the engine overheats?"):
        st.session_state.example_q = "What should I do if the engine overheats?"
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔎 View Retrieved Sources"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"**Source {idx+1}**")
                        st.markdown(f"**Document:** {src.get('document', 'Unknown')} | **Page:** {src.get('page', 1)} | **Relevance:** {src.get('relevance', 0.0)}")
                        st.markdown(f"*{src.get('text', '')}*")
                        st.markdown("---")

    prompt = st.chat_input("Ask a question about the automobile documents...")
    if "example_q" in st.session_state and st.session_state.example_q:
        prompt = st.session_state.example_q
        del st.session_state.example_q

    if prompt:
        # Extract prior conversation history (last 4 messages) before appending current turn
        hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-4:]]
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag_pipeline = RAGPipeline(
                        retriever=retriever,
                        llm_provider=st.session_state.llm_provider,
                        temperature=st.session_state.temperature
                    )
                    
                    answer, sources, _ = rag_pipeline.answer_question(
                        query=prompt, 
                        top_k=st.session_state.top_k, 
                        filter_document_name=filter_doc,
                        conversation_history=hist
                    )
                    
                    st.markdown(answer)
                    
                    if sources:
                        st.markdown("**Sources:**")
                        seen_citations = set()
                        for src in sources:
                            doc_name = src.get("document", "Unknown")
                            page_num = src.get("page", 1)
                            cite_label = f"📄 {doc_name} — Page {page_num}"
                            if cite_label not in seen_citations:
                                seen_citations.add(cite_label)
                                st.markdown(cite_label)
                            
                        with st.expander("🔎 View Retrieved Sources"):
                            for idx, src in enumerate(sources):
                                st.markdown(f"**Source {idx+1}**")
                                st.markdown(f"**Document:** {src.get('document', 'Unknown')} | **Page:** {src.get('page', 1)} | **Relevance:** {src.get('relevance', 0.0)}")
                                st.markdown(f"*{src.get('text', '')}*")
                                st.markdown("---")
                                
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                except Exception as e:
                    st.error(f"Error: {e}")

# ----------------- ⚖ COMPARE -----------------
elif nav_option == "⚖ Compare":
    st.title("⚖ Compare Vehicles")
    
    docs = chroma_store.get_all_documents_metadata()
    if len(docs) < 2:
        st.warning("Please upload at least two documents to compare.")
    else:
        doc_names = [d["name"] for d in docs]
        selected_docs = st.multiselect("Select documents to compare:", doc_names)
        
        compare_query = st.text_input("What would you like to compare? (e.g., 'Compare their engine specifications')")
        
        if st.button("Compare") and len(selected_docs) >= 2 and compare_query:
            with st.spinner("Analyzing..."):
                try:
                    rag_pipeline = RAGPipeline(
                        retriever=retriever,
                        llm_provider=st.session_state.llm_provider,
                        temperature=st.session_state.temperature
                    )
                    
                    answer, sources = rag_pipeline.compare_vehicles(compare_query, selected_docs, top_k_per_doc=st.session_state.top_k)
                    
                    st.markdown(answer)
                    
                    if sources:
                        st.markdown("**Sources:**")
                        seen_citations = set()
                        for src in sources:
                            doc_name = src.get("document", "Unknown")
                            page_num = src.get("page", 1)
                            cite_label = f"📄 {doc_name} — Page {page_num}"
                            if cite_label not in seen_citations:
                                seen_citations.add(cite_label)
                                st.markdown(cite_label)
                except Exception as e:
                    st.error(f"Error: {e}")

# ----------------- 📚 KNOWLEDGE BASE -----------------
elif nav_option == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")
    docs = chroma_store.get_all_documents_metadata()
    if not docs:
        st.info("Knowledge base is empty.")
    else:
        for doc in docs:
            st.markdown(f"### 📄 {doc['name']}")
            st.markdown(f"- **Pages indexed:** {doc['pages']}")
            st.markdown(f"- **Total chunks:** {doc['chunks']}")
            st.markdown("---")

# ----------------- ⚙ SETTINGS -----------------
elif nav_option == "⚙ Settings":
    st.title("⚙ Settings")
    
    st.subheader("LLM Provider")
    st.session_state.llm_provider = st.selectbox(
        "Select Provider", 
        ["gemini", "openai", "claude"],
        index=["gemini", "openai", "claude"].index(st.session_state.llm_provider)
    )
    
    st.subheader("Retrieval Settings")
    st.session_state.top_k = st.slider("Top K (Chunks to retrieve)", 1, 15, st.session_state.top_k)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature)
    
    st.subheader("System Status")
    st.success("Vector Database: ChromaDB (● Active)")
