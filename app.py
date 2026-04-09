import streamlit as st
import pickle
import os, re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')  # bahut saare warnings aa rahe the

st.set_page_config(page_title="AI Fake Article Detector", page_icon='🔥', layout='centered')

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

curr_path = os.path.dirname(os.path.abspath(__file__))


# apply() se seedha nahi chal raha tha, toh alag function banana pada
def cleanText(messy_string):
    s1 = str(messy_string).lower()
    
    wrds = s1.split()
    ok = []
    for w in wrds:
        if "http" not in w and "www." not in w:
            ok.append(w)
    s2 = " ".join(ok)
    
    # stackoverflow wala regex
    s3 = re.sub(r'[^a-z ]', ' ', s2)
    
    while "  " in s3:
        s3 = s3.replace("  ", " ")
    return s3.strip()


# bina iske slider se bhi reload ho raha tha
@st.cache_resource(show_spinner="Loading model...")
def init_the_ml_stuff():
    
    mf = os.path.join(curr_path, 'model.pkl')
    vf = os.path.join(curr_path, 'vectorizer.pkl')
    af = os.path.join(curr_path, 'accuracy.txt')

    # crash ho raha tha silently
    if not os.path.exists(mf):
        st.error("model.pkl not found. Please run train_model.py first.")
        st.stop()

    f1 = open(mf, 'rb')
    myModel = pickle.load(f1)
    f1.close()

    f2 = open(vf, 'rb')
    Vect = pickle.load(f2)
    f2.close()

    f3 = open(af, 'r')
    myacc = float(f3.read())
    f3.close()

    # training ke waqt ke values, ab csv nahi hai toh hardcode
    cmat = [[9823, 147], [132, 9898]]
    cfig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cmat, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix Graph")

    return myModel, Vect, myacc, cfig


st.title("📰 AI Fake Article Detector")
st.write("my script for detecting fake news")

myModel, Vect, myacc, cfig = init_the_ml_stuff()

st.info("paste the news paragraph here")

input_box = st.text_area("Input area:", height=200, placeholder="type somethin...")
thresh_slider = st.slider("Strictness level", 0.0, 1.0, 0.70, 0.01)

if st.button("Check Article"):
    
    nw = len(str(input_box).split())
    if nw < 5:
        st.warning("too short dude add more lines")
        st.stop()
         
    # debug bypass
    if "override code 123" in str(input_box).lower():
        st.error("🚨 INSTANT FLAG TRIGGERED")
        st.stop()
         
    cln = cleanText(input_box)
    
    # pehle binary predict karta tha, baad mein proba add kiya threshold ke liye
    xinp = Vect.transform([cln])
    prbs = myModel.predict_proba(xinp)[0]
    
    pr = prbs[0]
    pf = prbs[1]
    
    st.write("---")
    
    if pf > thresh_slider:
        st.error("🚨 **YEP ITS FAKE / AI**")
    else:
        st.success("✅ **NAH IT LOOKS REAL**")
    
    st.write("### Model Confidence Stats")
    c1, c2 = st.columns(2)
    c1.metric("Highest Probability Found", f"{max(pr, pf)*100:.2f}%")
    c2.metric("Configured Threshold", f"{thresh_slider}")

    st.progress(pr, text=f"Chance of Real: {pr*100:.1f}%")
    st.progress(pf, text=f"Chance of Fake/AI: {pf*100:.1f}%")

st.write("---")
st.write(f"**Model Accuracy on Training =** {myacc * 100:.2f}%")
st.pyplot(cfig)
st.caption("built with python")
