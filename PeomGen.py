import streamlit as st
import tensorflow as tf
import numpy as np
import pickle


# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r'C:\Users\zaina\Downloads\poetry_model.h5')


# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as f:
        return pickle.load(f)


# Poetry generation function
def generate_poetry(seed_text, next_words=20, model=None, tokenizer=None):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=model.input_shape[1] - 1,
                                                                   padding='pre')
        predicted_probabilities = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probabilities)
        predicted_word = tokenizer.index_word.get(predicted_word_index, "")
        seed_text += ' ' + predicted_word
    return seed_text


# Enhanced CSS with better readability
st.markdown("""
    <style>
        /* Global Styles */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #1a1b26, #2d2d44);
            color: #ffffff;
            padding: 2rem;
            min-height: 100vh;
        }

        /* Hide Streamlit's default header */
        header {
            visibility: hidden;
        }

        /* Title Styling */
        .title-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2.5rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .main-title {
            background: linear-gradient(90deg, #ff9b9b, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            letter-spacing: 1px;
        }

        .subtitle {
            color: #e0e0e0;
            font-size: 1.2rem;
            font-weight: 400;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }

        /* Input Area Styling */
        .stTextArea textarea {
            background-color: rgba(255, 255, 255, 0.95);
            color: #1a1b26;
            border-radius: 12px;
            border: 2px solid rgba(255, 255, 255, 0.1);
            padding: 1.2rem;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Labels and Headers */
        .stTextArea label, .stSlider label {
            color: #ffffff !important;
            font-size: 1.1rem !important;
            font-weight: 500 !important;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
            margin-bottom: 0.5rem !important;
        }

        /* Slider Styling */
        .stSlider {
            padding: 1.5rem 0;
        }

        .stSlider [data-baseweb="slider"] {
            margin-top: 1rem;
        }

        /* Button Styling */
        .stButton button {
            background: linear-gradient(90deg, #ff6b6b, #ffd700);
            color: #ffffff;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.2rem;
            width: 100%;
            margin-top: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
        }

        /* Generated Poetry Output */
        .poetry-output {
            background: rgba(255, 255, 255, 0.95);
            color: #1a1b26;
            padding: 2rem;
            border-radius: 12px;
            margin-top: 1.5rem;
            font-size: 1.2rem;
            line-height: 1.8;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            white-space: pre-line;
        }
    </style>
""", unsafe_allow_html=True)


def main():
    # Load model and tokenizer
    model = load_model()
    tokenizer = load_tokenizer()

    # Main Layout
    st.markdown("""
        <div class="title-container">
            <h1 class="main-title">✨ Roman Urdu Poetry Generator</h1>
            <p class="subtitle">Craft beautiful poetry with artificial intelligence</p>
        </div>
    """, unsafe_allow_html=True)

    # Input area
    user_input = st.text_area(
        "Enter your starting words:",
        placeholder="Start writing here...",
        height=150
    )

    # Poetry length slider
    poetry_length = st.slider(
        "Poetry Length",
        min_value=50,
        max_value=200,
        value=100
    )

    # Generate button
    if st.button("✨ Generate Poetry"):
        if user_input:
            with st.spinner("✨ Creating your masterpiece..."):
                generated_poetry = generate_poetry(user_input, poetry_length, model, tokenizer)
                st.markdown(f'<div class="poetry-output">{generated_poetry}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to generate poetry.")


if __name__ == "__main__":
    main()