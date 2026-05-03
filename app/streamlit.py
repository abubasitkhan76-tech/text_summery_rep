import streamlit as st
import requests

st.set_page_config(page_title="BART Text Summarizer", page_icon="📝")

st.title("📝 AI Text Summarizer")
st.markdown("This app uses your **BART model** running on FastAPI (D Drive).")

# Input text box
input_text = st.text_area("Paste your long text here:", height=300)

if st.button("Summarize"):
    if input_text:
        with st.spinner("Processing..."):
            # Connect to your FastAPI server
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/summarize",
                    json={"text": input_text, "max_length": 130, "min_length": 30}
                )
                
                if response.status_code == 200:
                    summary = response.json().get("summary")
                    st.subheader("Summary:")
                    st.success(summary)
                else:
                    st.error("Error: Could not get summary from API.")
            except Exception as e:
                st.error(f"Connection Failed: Is your FastAPI running? {e}")
    else:
        st.warning("Please paste some text first!")
