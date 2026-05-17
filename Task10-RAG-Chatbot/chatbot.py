"""
Task 10: Context-Aware RAG Chatbot with Groq
Using FakeEmbeddings (matching ingestion script)
"""

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="RAG Chatbot - Developers Hub", page_icon="🤖", layout="wide")

st.title("🤖 Context-Aware RAG Chatbot")
st.markdown("Powered by **Groq's Llama 3.3 70B** | Remembers conversations | Retrieves from documents")

with st.sidebar:
    st.header("📚 About")
    st.markdown("""
    **Try asking:**
    - "What services do you offer?"
    - "Tell me about the internship program"
    - "What technologies do you use?"
    - "Who are your partners?"
    """)
    
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API: Connected")
    else:
        st.error("❌ Groq API: Key not found")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "chain" not in st.session_state:
    with st.spinner("⚡ Loading chatbot..."):
        try:
            # Use FakeEmbeddings to match the ingestion script
            embeddings = FakeEmbeddings(size=384)
            
            # Load vector store
            vectorstore = Chroma(
                persist_directory='vector_store',
                embedding_function=embeddings
            )
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Initialize Groq LLM
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=512
            )
            
            # Create conversation chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                verbose=False
            )
            
            st.session_state.chain = chain
            st.success("✅ Chatbot ready! Powered by Groq Llama 3.3 70B")
            
        except Exception as e:
            st.error(f"Error loading chatbot: {str(e)}")
            st.info("Make sure you've run 'python ingest_documents.py' first")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📖 Sources"):
                for source in message["sources"]:
                    st.text(source[:200] + "...")

# Chat input
if prompt := st.chat_input("Ask me anything about Developers Hub Corporation..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("⚡ Thinking..."):
            try:
                if "chain" in st.session_state:
                    response = st.session_state.chain.invoke({"question": prompt})
                    answer = response["answer"]
                    sources = []
                    
                    if "source_documents" in response:
                        for doc in response["source_documents"][:2]:
                            sources.append(doc.page_content[:300])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}:** {source}...")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error("Chatbot not initialized. Please check your setup.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.sidebar.divider()
if "chain" in st.session_state:
    st.sidebar.success("✅ Chatbot: Online")
    st.sidebar.info(f"📝 Messages: {len(st.session_state.messages)}")