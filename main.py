# Module 1: Import necessary packages
import streamlit as st

# Set page configuration - must be the first Streamlit command
page_icon = ":metro:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "wide"
page_title = "Fake News Detection"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import warnings
import streamlit_lottie
warnings.filterwarnings("ignore")

# Module 2: Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("fake_or_real_news.csv")
    data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
    return data

# Module 3: Select Vectorizer and Classifier
def select_model():
    # Apply a modern, visually appealing color palette
    st.markdown("""
    <style>
    .stApp {
        background-color: #f7ede2 !important; /* light creamy brown */
        color: #5e4636 !important; /* soft brown */
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #e3f0fa !important; /* very light blue */
        color: #5e4636 !important;
        border: 1px solid #bca18d !important;
    }
    .stButton > button {
        background-color: #7ec4cf !important; /* light blue */
        color: #5e4636 !important;
        border-radius: 6px;
        border: none;
        font-weight: bold;
        transition: background 0.2s, color 0.2s;
    }
    .stButton > button:hover {
        background-color: #f7a1c4 !important; /* pinkish */
        color: #5e4636 !important;
    }
    .stSidebarContent {
        background-color: #f7ede2 !important;
        color: #5e4636 !important;
    }
    ::selection {
        background: #b3e0ff !important; /* light blue highlight */
        color: #5e4636 !important;
    }
    .stRadio > div > label {
        color: #5e4636 !important;
    }
    .stSelectbox > div > div {
        background-color: #e3f0fa !important;
        color: #5e4636 !important;
    }
    .stMarkdown, .stTitle, .stHeader, .stSubheader, .stText, .stCaption {
        color: #5e4636 !important;
    }
    .custom-label {
        font-size: 1.2rem;
        font-weight: bold;
        color: #5e4636 !important;
        margin-bottom: 0.5rem;
    }
    /* Make the hamburger menu (sidebar collapse button) more visible */
    [data-testid="collapsedControl"] svg {
        color: #3b4a6b !important; /* dark blue for contrast */
        width: 2.2rem !important;
        height: 2.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    vectorizer_type = st.sidebar.selectbox("Select Vectorizer", ["TF-IDF", "Bag of Words"])
    classifier_type = st.sidebar.selectbox("Select Classifier", ["Linear SVM", "Naive Bayes"])
    
    vectorizer = None
    if vectorizer_type == "TF-IDF":
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    elif vectorizer_type == "Bag of Words":
        vectorizer = CountVectorizer(stop_words='english', max_df=0.7)
    
    classifier = None
    if classifier_type == "Linear SVM":
        classifier = LinearSVC()
    elif classifier_type == "Naive Bayes":
        classifier = MultinomialNB()
    
    return vectorizer, classifier

# Module 4: Train the model
def train_model(data, vectorizer, classifier):
    x_vectorized = vectorizer.fit_transform(data['text'])
    clf = classifier.fit(x_vectorized, data['fake'])
    return clf

# Module 5: Streamlit app
def main():
    # Streamlit app
    st.title(page_title + " " + page_icon)
    # st.lottie("https://lottie.host/bd0c4818-c5a6-4e42-b407-746bc448c2c7/ipVUdgFncO.json", width=200, height=200)

    # --- HIDE STREAMLIT STYLE ---
    hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Load data
    data = load_data()
    
    # Select vectorizer and classifier
    vectorizer, classifier = select_model()
    
    # Text input for user to input news article
    st.markdown(
        """
        <style>
        .custom-label {
            font-size: 1.2rem;
            font-weight: bold;
            color: #5e4636;
            margin-bottom: 0.5rem;
        }
        </style>
        <div class=\"custom-label\">Enter your news article here:</div>
        """,
        unsafe_allow_html=True
    )
    user_input = st.text_area(" ", key="news_input")
    
    # When user submits the input
    if st.button("Check"):
        # Train the model
        clf = train_model(data, vectorizer, classifier)
        
        # Vectorize the user input
        input_vectorized = vectorizer.transform([user_input])
        
        # Predict the label of the input
        prediction = clf.predict(input_vectorized)
        
        # Convert prediction to integer for interpretation
        result = int(prediction[0])
        
        # Display the result with high visibility
        if result == 1:
            st.markdown(
                """
                <div style='background:#ff5252; color:white; font-size:1.3rem; font-weight:bold; border-radius:8px; padding:1rem; margin-top:1rem; text-align:center; box-shadow:0 2px 8px #ff525233;'>
                🚨 This news article is likely fake!
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style='background:#43a047; color:white; font-size:1.3rem; font-weight:bold; border-radius:8px; padding:1rem; margin-top:1rem; text-align:center; box-shadow:0 2px 8px #43a04733;'>
                ✅ This news article seems to be real.
                </div>
                """,
                unsafe_allow_html=True
            )

    # Contact Us section in the sidebar (fixed to bottom)
    st.sidebar.markdown("""
    <style>
    .sidebar-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 17rem;
        padding-bottom: 1.5rem;
        padding-left: 1.5rem;
        background: transparent;
        z-index: 9999;
    }
    @media (max-width: 600px) {
        .sidebar-footer { width: 100vw; left: 0; padding-left: 0.5rem; }
    }
    </style>
    <div class='sidebar-footer'>
        <hr style='border:1px solid #fff; margin: 0.5rem 0;'>
        <div style='line-height:1.8; color:#fff; font-weight:600;'>
        🌐 <b>Website:</b> <a href='https://www.codepirates.com' target='_blank' style='color:#fff;'>www.codepirates.com</a><br>
        📸 <b>Instagram:</b> <a href='https://instagram.com/codepirates' target='_blank' style='color:#fff;'>@codepirates</a><br>
        🐦 <b>Twitter:</b> <a href='https://twitter.com/codepirates' target='_blank' style='color:#fff;'>@codepirates</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()

# Custom footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #f7ede2;
        color: #5e4636;
        text-align: center;
        font-size: 1rem;
        padding: 0.5rem 0 0.2rem 0;
        border-top: 1px solid #e0cfc2;
        z-index: 100;
    }
    </style>
    <div class="footer">
        © 2025 Codeπrates. All rights reserved.<br>
        Akrant Debnath, Amogh Varsha K, Anish Sawhney, Dikshant Sharma
    </div>
    """,
    unsafe_allow_html=True
)

##run with command streamlit run main.py --client.showErrorDetails=false to remove cache error message on streamlit interface
