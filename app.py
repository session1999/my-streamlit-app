import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

st.set_page_config(page_title="Live Web Agent", layout="centered")

st.title("🌍 The Live Internet Agent")
st.write("Ask me anything about current events. I will browse the web to find the answer.")

# --- GET API KEY FROM SECRETS ---
try:
    # Try to get API key from Streamlit Secrets
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception as e:
    st.error("⚠️ Groq API Key not found in Secrets. Please add it in the Streamlit Cloud dashboard.")
    st.info("Go to your app → ⋮ (three dots) → Settings → Secrets → Add:")
    st.code('GROQ_API_KEY = "your-groq-api-key-here"', language="toml")
    st.stop()

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
        st.success("🌐 Web Search is **ON** - I'll search the internet for answers")
    else:
        st.info("💬 Web Search is **OFF** - I'll answer from my knowledge only")
    
    st.divider()
    st.caption("🔑 API Key is securely stored in Secrets")

# --- 2. THE MEMORY VAULT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. DRAW THE CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. THE CORE AGENTIC LOOP ---
if user_query := st.chat_input("Ask about today's news..."):
    
    # A. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # B. Initialize the LangGraph Agent Engine with secret key
    llm = ChatGroq(
        temperature=0, 
        model_name=model_option,
        api_key=GROQ_API_KEY  # Using the secret from above
    )
    
    # Create the search tool
    web_tool = DuckDuckGoSearchRun()
    
    # Tool toggle logic
    if use_search:
        active_tools = [web_tool]
        tool_status = "with web search enabled"
    else:
        active_tools = []
        tool_status = "with web search disabled (knowledge only)"
    
    # System prompts
    if use_search:
        sys_prompt = """You are a live research assistant. 
        You have access to a web search tool. 
        You MUST use the web search tool to find current information before answering.
        
        When you receive a question:
        1. First, use the web search tool to find relevant information
        2. Then, synthesize the results into a clear, accurate answer
        3. Always cite your sources when possible"""
    else:
        sys_prompt = """You are a helpful AI assistant.
        You do not have access to web search in this mode.
        Answer questions based on your training knowledge only.
        If you don't know something, simply say so."""
    
    # Create agent
    agent = create_react_agent(
        llm, 
        tools=active_tools,
        prompt=sys_prompt,
        checkpointer=MemorySaver()
    )

    # Prepare messages
    langgraph_messages = [SystemMessage(content=sys_prompt)]
    for m in st.session_state.messages:
        if m["role"] == "user":
            langgraph_messages.append(HumanMessage(content=m["content"]))
        else:
            langgraph_messages.append(AIMessage(content=m["content"]))

    # Execute agent
    with st.chat_message("assistant"):
        with st.spinner(f"🤖 Thinking {tool_status}..."):
            try:
                result_state = agent.invoke({
                    "messages": langgraph_messages
                }, config={"configurable": {"thread_id": "1"}})
                
                final_messages = result_state["messages"]
                bot_answer = final_messages[-1].content if final_messages else "No answer found."
                
            except Exception as e:
                if use_search:
                    try:
                        search_result = web_tool.run(user_query)
                        bot_answer = f"**Search Results:**\n\n{search_result}"
                    except:
                        bot_answer = f"I encountered an error: {str(e)}"
                else:
                    bot_answer = f"I encountered an error: {str(e)}"
        
        st.markdown(bot_answer)
    
    # Save response
    st.session_state.messages.append({"role": "assistant", "content": bot_answer})
