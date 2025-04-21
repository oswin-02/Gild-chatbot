import streamlit as st
from openai import OpenAI
import time
import re

placeholderstr = "Please input your command"
user_name = "oswin-02"
user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def main():
    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
            },
        page_icon="img/favicon.ico"
    )

    # Show title and description.
    st.title(f"💬 {user_name}'s Chatbot")

    with st.sidebar:
        selected_lang = st.selectbox("Language", ["English", "繁體中文"], index=1)
        if 'lang_setting' in st.session_state:
            lang_setting = st.session_state['lang_setting']
        else:
            lang_setting = selected_lang
            st.session_state['lang_setting'] = lang_setting

        st_c_1 = st.container(border=True)
        with st_c_1:
            st.image("https://www.w3schools.com/howto/img_avatar.png")

    st_c_chat = st.container(border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    def generate_response(prompt):
        prompt = prompt.lower().strip()

        # Keywords for each category
        sectors = {
            "defense": ["lockheed martin", "raytheon", "northrop", "general dynamics", "boeing", "huntington", "tdg", "curtiss", "l3harris", "axon"],
            "technology": ["apple", "microsoft", "google", "meta", "nvidia", "oracle", "ibm", "amd", "intel", "salesforce"],
            "agriculture": ["adm", "bunge", "deere", "cf industries", "mosaic", "fmc", "tyson", "corteva", "agco", "scotts"],
            "oil and gas": ["exxon", "chevron", "conocophillips", "eog", "occidental", "hess", "devon", "kinder morgan"]
        }

        # Project overview
        if any(word in prompt for word in ["what is this project", "goal", "aim", "purpose"]):
            return "The goal of this project is to find correlations between Ukraine-Russia war events and stock movements of specific companies to help predict future trends."

        # Dataset explanation
        elif "dataset" in prompt or "data" in prompt:
            return ("We use data from two sources:\n"
                    "- War-related news and timelines from reliable sources.\n"
                    "- Stock prices of companies across four sectors: Defense, Technology, Agriculture, and Oil & Gas.")

        # Sector/company detection
        for sector, keywords in sectors.items():
            if any(k in prompt for k in keywords) or sector in prompt:
                return f"Our findings suggest that {sector.title()} sector stocks respond differently to war sentiment. Want to hear about the correlation?"

            # Sentiment analysis method
            elif "sentiment" in prompt or "bert" in prompt:
                return ("We use a fine-tuned DistilBERT model to classify sentiment of war-related news. "
                        "This gives us a time-series of sentiment scores to correlate with stock price trends.")

            # Correlation method
            elif "correlation" in prompt or "relationship" in prompt:
                return ("We compare sentiment trends with stock price trends using correlation analysis. "
                        "This helps us see which sectors respond positively or negatively to changes in public sentiment.")

            # Text processing method
            elif "word cloud" in prompt or "text processing" in prompt or "cleaning" in prompt:
                return ("We extract war-related content from PDFs using PyMuPDF, clean it with stopword and noise filtering, "
                        "and visualize important terms using TF-IDF word clouds.")

            # Prediction method
            elif "predict" in prompt or "future" in prompt:
                return ("Based on past sentiment-stock relationships, we explore how current sentiment may suggest potential movements "
                        "in sectors like tech or oil.")

            # Conclusion
            elif "conclusion" in prompt or "result" in prompt:
                return ("Our conclusion:\n"
                        "- 📈 Tech stocks rise with positive sentiment.\n"
                        "- 🛢️ Oil & Gas stocks tend to rise during negative sentiment.\n"
                        "- 🚜 Agriculture and 🛡️ Defense show no consistent pattern.")

            # Greetings or fallback
            elif re.search(r"\b(hi|hello|hey|what can you do|who are you)\b", prompt):
                return "Hi! I'm your Ukraine-Russia Stock Correlation Assistant. Ask me about stock sectors, sentiment analysis, or what we found in our project."

            else:
                return "Try asking about our sentiment method, stock sectors, dataset, or project findings!"
            
    # Chat function section (timing included inside function)
    def chat(prompt: str):
        st_c_chat.chat_message("user",avatar=user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = generate_response(prompt)
        # response = f"You type: {prompt}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))

    
    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

if __name__ == "__main__":
    main()
