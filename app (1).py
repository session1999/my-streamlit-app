import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

st.set_page_config(page_title="Live Web Agent", layout="centered")

st.title("🌍 The Live Internet Agent")
st.write("Ask me anything about current events. I will browse the web to find the answer.")

# --- 1. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ System Config")
    user_api_key = st.text_input("Groq API Key:", type="password")
    st.info("Equipped with: DuckDuckGo Web Search Tool")
    
    st.divider()
    st.header("🤖 Model Selection")
    model_option = st.selectbox(
        "Choose a model:",
        [
            "llama-3.3-70b-versatile",  # Latest Llama 3.3
            "llama-3.1-8b-instant",     # Fast Llama 3.1 8B
            "mixtral-8x7b-32768",       # Mixtral
            "gemma2-9b-it"              # Google's Gemma 2
        ],
        index=0
    )
    
    st.divider()
    st.header("🛠️ Tool Control")
    # NEW: Toggle switch for web search
    use_search = st.toggle("Enable Web Search", value=True)
    if use_search:
        st.success("🌐 Web Search is **ON** - I'll search the internet for answers")
    else:
        st.info("💬 Web Search is **OFF** - I'll answer from my knowledge only")

# --- 2. THE MEMORY VAULT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. DRAW THE CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. THE CORE AGENTIC LOOP ---
if user_query := st.chat_input("Ask about today's news..."):
   
    if not user_api_key:
        st.error("Please enter your API Key in the sidebar.")
    else:
        # A. Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # B. Initialize the LangGraph Agent Engine
        llm = ChatGroq(
            temperature=0, 
            model_name=model_option,
            api_key=user_api_key
        )
        
        # Create the search tool
        web_tool = DuckDuckGoSearchRun()
        
        # NEW: Tool toggle logic - set active tools based on toggle
        if use_search:
            active_tools = [web_tool]
            tool_status = "with web search enabled"
        else:
            active_tools = []  # Empty list = no tools, just chat
            tool_status = "with web search disabled (knowledge only)"
        
        # Define system prompt - modified based on tool availability
        if use_search:
            sys_prompt = """You are a live research assistant. 
            You have access to a web search tool. 
            You MUST use the web search tool to find current information before answering.
            
            When you receive a question:
            1. First, use the web search tool to find relevant information
            2. Then, synthesize the results into a clear, accurate answer
            3. Always cite your sources when possible
            
            Remember: Always search first, then answer based on the search results."""
        else:
            sys_prompt = """You are a helpful AI assistant.
            You do not have access to web search in this mode.
            Answer questions based on your training knowledge only.
            If you don't know something, simply say so."""
        
        # Create the agent with active_tools (which may be empty)
        agent = create_react_agent(
            llm, 
            tools=active_tools,  # This can be [] if search is disabled
            prompt=sys_prompt,
            checkpointer=MemorySaver()
        )

        # C. Translate Streamlit Memory -> LangGraph Memory
        langgraph_messages = []
        langgraph_messages.append(SystemMessage(content=sys_prompt))
        
        for m in st.session_state.messages:
            if m["role"] == "user":
                langgraph_messages.append(HumanMessage(content=m["content"]))
            else:
                langgraph_messages.append(AIMessage(content=m["content"]))

        # D. Execute the Agent
        with st.chat_message("assistant"):
            with st.spinner(f"🤖 Thinking {tool_status}..."):
                try:
                    result_state = agent.invoke({
                        "messages": langgraph_messages
                    }, config={"configurable": {"thread_id": "1"}})
                    
                    final_messages = result_state["messages"]
                    bot_answer = final_messages[-1].content if final_messages else "No answer found."
                    
                except Exception as e:
                    # Fallback to simple search ONLY if search is enabled
                    if use_search:
                        try:
                            search_result = web_tool.run(user_query)
                            bot_answer = f"**Search Results:**\n\n{search_result}"
                        except:
                            bot_answer = f"I encountered an error: {str(e)}"
                    else:
                        bot_answer = f"I encountered an error: {str(e)}"
            
            st.markdown(bot_answer)
           
        # E. Save the final answer
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
