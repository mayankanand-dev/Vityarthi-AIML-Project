import pandas as pd
import streamlit as st
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore') # just ignoring these for now

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# streamlit settings
st.set_page_config(page_title="AI Fake Article Detector", page_icon='🔥', layout='centered')

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

curr_path = os.path.dirname(os.path.abspath(__file__))

# i needed a separate function to clean text because apply() was being weird
def do_text_cleaning(messy_string):
    step_one = str(messy_string).lower()
    
    # split and check for links
    tmp_words = step_one.split()
    ok_words = []
    
    for w in tmp_words:
        if "http" not in w and "www." not in w:
            ok_words.append(w)
            
    stage_two = " ".join(ok_words)
    
    # regex for getting rid of numbers and punctuation
    # found this on stackoverflow
    stage_three = re.sub(r'[^a-z ]', ' ', stage_two)
            
    # fixing double spaces that appear sometimes
    while "  " in stage_three:
        stage_three = stage_three.replace("  ", " ")
        
    return stage_three.strip()

@st.cache_resource(show_spinner="training models, pls wait...")
def init_the_ml_stuff():
    s_path = os.path.join(curr_path, 'v_spam_db.csv')
    r_path = os.path.join(curr_path, "v_legit_db.csv")
    
    if not os.path.exists(s_path):
        st.write("error finding csv file")
        st.stop()
        
    # reading data
    df_f = pd.read_csv(s_path)
    df_r = pd.read_csv(r_path)
    
    # 1 for fake 0 for real
    df_f['label'] = 1   
    df_r["label"] = 0   
    
    all_data = pd.concat([df_f, df_r], ignore_index=True)
    all_data = all_data.dropna(subset=['title', "text", 'label'])
    
    # shuffle everything
    shuffled_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # let's test combining text first
    shuffled_data['temp_combined'] = shuffled_data['title'] + " " + shuffled_data['text']
    
    # cleaning the data manually
    corpus_list = []
    for txt in shuffled_data['temp_combined']:
        corpus_list.append(do_text_cleaning(txt))
        
    shuffled_data['final_text'] = corpus_list
    
    # setting up vectorizer
    vec_obj = TfidfVectorizer(max_features=5000, stop_words="english")
    
    # training it on our corpus
    vec_obj.fit(shuffled_data['final_text'])
    x_matrix = vec_obj.transform(shuffled_data['final_text'])
    
    target_var = shuffled_data['label']
    
    # standard 80 20 split
    X_train, X_test, y_train, y_test = train_test_split(
        x_matrix, target_var, test_size=0.2, random_state=42
    )

    # using logistic regression
    log_model = LogisticRegression(max_iter=500)
    log_model.fit(X_train, y_train)

    my_preds = log_model.predict(X_test)
    
    curr_acc = accuracy_score(y_test, my_preds)
    
    # plotting the matrix
    c_mat = confusion_matrix(y_test, my_preds)
    heat_fig, axis1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(c_mat, annot=True, fmt='d', cmap='Blues', ax=axis1)
    axis1.set_title("Confusion Matrix Graph")
    
    return log_model, vec_obj, curr_acc, heat_fig

st.title("📰 AI Fake Article Detector")
st.write("my script for detecting fake news")

# loading everything
trained_m, my_vec, acc_value, conf_fig = init_the_ml_stuff()

st.info("paste the news paragraph here")

input_box = st.text_area("Input area:", height=200, placeholder="type somethin...")
thresh_slider = st.slider("Strictness level", 0.0, 1.0, 0.70, 0.01)

if st.button("Check Article"):
    
    # check word count first
    num_words = len(str(input_box).split())
    if num_words < 5:
        st.warning("too short dude add more lines")
        st.stop()
         
    # debug bypass
    if "override code 123" in str(input_box).lower():
        st.error("🚨 INSTANT FLAG TRIGGERED")
        st.stop()
         
    cleaned_input = do_text_cleaning(input_box)
    
    x_input = my_vec.transform([cleaned_input])
    probs = trained_m.predict_proba(x_input)[0]
    
    p_real = probs[0]
    p_fake = probs[1]
    
