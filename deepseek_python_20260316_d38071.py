import streamlit as st
import subprocess
import sys

st.title("Package Test App")

# Check what's installed
st.write("Checking installed packages...")

try:
    import langchain_groq
    st.success("✅ langchain-groq is installed!")
except ImportError:
    st.error("❌ langchain-groq NOT found")
    # Try to install it
    st.write("Attempting to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-groq"])
    st.write("Installation attempted. Refresh the page.")

try:
    from langchain_groq import ChatGroq
    st.success("✅ ChatGroq imported successfully!")
except Exception as e:
    st.error(f"Import error: {e}")