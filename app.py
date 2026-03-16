import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS

st.set_page_config(page_title="Live Web Agent", layout="centered")

st.title("🌍 The Live Internet Agent")
st.write("Ask me anything about current events. I will browse the web to find the answer.")

# --- GET API KEY FROM SECRETS ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error("⚠️ Groq API Key not found in Secrets. Please add it in the Streamlit Cloud dashboard.")
    st.info("Go to your app → ⋮ (three dots) → Settings → Secrets → Add:")
    st.code('GROQ_API_KEY = "your-groq-api-key-here"', language="toml")
    st.stop()

# Simple web search function using direct DDGS
def simple_web_search(query):
    """Simple web search using DuckDuckGo directly"""
    try:
        with DDGS() as ddgs:
            results = []
            # Get text search results
            for r in ddgs.text(query, max_results=3):
                results.append(f"• {r.get('title', '')}: {r.get('body', '')[:200]}...")
            
            if results:
                return "Here's what I found:\n\n" + "\n\n".join(results)
            else:
                return f"I couldn't find specific results for '{query}'."
    except Exception as e:
        return f"Search encountered an issue: {str(e)}"

# --- 1. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🤖 Model Selection")
    model_option = st.selectbox(
        "Choose a model:",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=0
    )
    
    st.divider()
    st.header("🛠️ Tool Control")
    use_search = st.toggle("Enable Web Search", value=True)
    if use_search:
        st.success("🌐 Web Search is **ON** - I'll search the internet")
    else:
        st.info("💬 Web Search is **OFF** - Knowledge only")
    
    st.divider()
    st.caption("🔑 API Key stored in Secrets")

# --- 2. MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. DISPLAY HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. MAIN LOOP ---
if user_query := st.chat_input("Ask about today's news..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Initialize LLM
    llm = ChatGroq(
        temperature=0.7, 
        model_name=model_option,
        groq_api_key=GROQ_API_KEY  # Note: parameter is groq_api_key, not api_key
    )
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # If search is enabled, get search results first
                if use_search:
                    search_results = simple_web_search(user_query)
                    
                    # Create prompt with search results
                    enhanced_prompt = f"""Search results: {search_results}

Based on the search results above, please answer this question: {user_query}

If the search results don't contain enough information, just answer based on your knowledge."""
                    
                    messages = [HumanMessage(content=enhanced_prompt)]
                    response = llm.invoke(messages)
                    bot_answer = response.content
                else:
                    # No search, just use the LLM directly
                    messages = [HumanMessage(content=user_query)]
                    response = llm.invoke(messages)
                    bot_answer = response.content
                    
            except Exception as e:
                bot_answer = f"I encountered an error: {str(e)}"
            
            st.markdown(bot_answer)
    
    # Save response
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
