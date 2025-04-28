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

        # Greeting / fallback
        if re.search(r"\b(hi|hello|hey|what can you do|who are you|help)\b", prompt):
            return ("Hello! 👋 I can explain the pages you see:\n"
                    "- 2D Visualization of Word Embeddings\n"
                    "- 3D Visualization of Word Embeddings\n"
                    "- Skip-Gram Word2Vec: With and Without Stopwords\n"
                    "- Comparison of Skip-gram and CBOW Word2Vec Models\n"
                    "Ask me about any of these!")

        # 2D Visualization page
        if "2d" in prompt or ("2d visualization" in prompt) or ("2d plot" in prompt):
            return ("📊 **2D Visualization of Word Embeddings**:\n"
                    "This page projects high-dimensional word vectors into two dimensions using **PCA** or **t-SNE**, "
                    "so you can visually explore relationships between words. Closer words tend to have similar meanings!")

        # 3D Visualization page
        if "3d" in prompt or ("3d visualization" in prompt) or ("3d plot" in prompt):
            return ("📈 **3D Visualization of Word Embeddings**:\n"
                    "Similar to the 2D version, but here the word vectors are projected into **three dimensions**, "
                    "allowing you to rotate and interact with the embedding space more richly!")

        # Skip-gram Word2Vec: With and Without Stopwords
        if "stopword" in prompt or "skip-gram with stopwords" in prompt or "skip-gram without stopwords" in prompt:
            return ("🛑 **Skip-Gram Word2Vec: With and Without Stopwords**:\n"
                    "This page trains two Skip-Gram Word2Vec models: one with stopwords included, and one without stopwords. "
                    "It helps show how common words (like 'the', 'is') can influence the quality of the learned embeddings.")

        # Comparison of Skip-gram and CBOW Word2Vec Models
        if "cbow" in prompt or "skip-gram" in prompt or ("comparison" in prompt and "word2vec" in prompt):
            return ("⚔️ **Comparison of Skip-gram and CBOW Word2Vec Models**:\n"
                    "This page compares two training strategies:\n"
                    "- **Skip-gram**: predicts context from a center word (good for rare words).\n"
                    "- **CBOW**: predicts a center word from its context (faster for large corpora).\n"
                    "You can see their differences by checking similar words and embedding vectors!")

        # If not recognized
        return ("I'm not sure what you mean. 🤔\n"
                "Try asking about:\n"
                "- 2D Visualization\n"
                "- 3D Visualization\n"
                "- Skip-Gram With/Without Stopwords\n"
                "- Comparison of Skip-gram and CBOW Models")

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
