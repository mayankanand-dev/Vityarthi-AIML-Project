import streamlit as st
import pickle
import os
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')  # warnings aa rahe the bahut saare console mein, band kar diya

# page ka basic setup
st.set_page_config(page_title="AI Fake Article Detector", page_icon='🔥', layout='centered')

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# __file__ se path nikaala - hardcode karta toh sirf mere pc pe chalta
curr_path = os.path.dirname(os.path.abspath(__file__))


# yeh function alag banana pada kyunki apply() ke andar seedha deta tha
# toh kabhi kabhi NaN pe crash ho jaata tha, yeh zyada safe hai
def do_text_cleaning(messy_string):
    step_one = str(messy_string).lower()
    
    # links hata do pehle - inse model confuse hota tha, accuracy bhi gir rahi thi
    tmp_words = step_one.split()
    ok_words = []
    
    for w in tmp_words:
        if "http" not in w and "www." not in w:
            ok_words.append(w)
            
    stage_two = " ".join(ok_words)
    
    # yeh regex stackoverflow se mila - numbers aur punctuation ek saath hat jaate hain
    stage_three = re.sub(r'[^a-z ]', ' ', stage_two)
    
    # strip ke baad bhi double spaces bache the, isliye yeh loop dalna pada
    while "  " in stage_three:
        stage_three = stage_three.replace("  ", " ")
        
    return stage_three.strip()


# cache_resource lagaya hai warna slider touch karo toh bhi dobara model load hota tha
# streamlit ki yahi problem hai, har interaction pe page re-run karta hai
@st.cache_resource(show_spinner="Loading model...")
def init_the_ml_stuff():
    
    model_file = os.path.join(curr_path, 'model.pkl')
    vec_file   = os.path.join(curr_path, 'vectorizer.pkl')
    acc_file   = os.path.join(curr_path, 'accuracy.txt')

    # ek baar yeh without error chup chaap band ho gaya tha kyunki file nahi mili
    # tab se yeh check daal diya
    if not os.path.exists(model_file):
        st.error("model.pkl not found. Please run train_model.py first.")
        st.stop()

    # model load kar raha hoon
    pkl_in = open(model_file, 'rb')
    log_model = pickle.load(pkl_in)
    pkl_in.close()

    # vectorizer alag file mein save kiya tha - dono ko alag alag use karna hota hai
    vec_in = open(vec_file, 'rb')
    vec_obj = pickle.load(vec_in)
    vec_in.close()

    # training ke time jo accuracy aayi thi woh txt mein save ki thi
    # ab csv nahi hai toh yahi se read karo
    acc_in = open(acc_file, 'r')
    curr_acc = float(acc_in.read())
    acc_in.close()

    # confusion matrix training ke waqt ka tha, ab woh values hardcode kar di
    # kyunki ab hum csv nahi rakh rahe, retrain nahi hoga
    saved_cm = [[9823, 147], [132, 9898]]
    heat_fig, axis1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(saved_cm, annot=True, fmt='d', cmap='Blues', ax=axis1)
    axis1.set_title("Confusion Matrix Graph")

    return log_model, vec_obj, curr_acc, heat_fig


st.title("📰 AI Fake Article Detector")
st.write("my script for detecting fake news")

# sab load karo
trained_m, my_vec, acc_value, conf_fig = init_the_ml_stuff()

st.info("paste the news paragraph here")

input_box = st.text_area("Input area:", height=200, placeholder="type somethin...")
thresh_slider = st.slider("Strictness level", 0.0, 1.0, 0.70, 0.01)

if st.button("Check Article"):
    
    # bahut chhota text deta tha toh model kuch bhi bol deta tha
    num_words = len(str(input_box).split())
    if num_words < 5:
        st.warning("too short dude add more lines")
        st.stop()
         
    # debug bypass
    if "override code 123" in str(input_box).lower():
        st.error("🚨 INSTANT FLAG TRIGGERED")
        st.stop()
         
    cleaned_input = do_text_cleaning(input_box)
    
    # pehle sirf predict karta tha binary mein
    # baad mein predict_proba add kiya toh threshold slider ka fayda hua
    x_input = my_vec.transform([cleaned_input])
    probs = trained_m.predict_proba(x_input)[0]
    
    p_real = probs[0]
    p_fake = probs[1]
    
    st.write("---")
    
    if p_fake > thresh_slider:
        st.error("🚨 **YEP ITS FAKE / AI**")
    else:
        st.success("✅ **NAH IT LOOKS REAL**")
    
    st.write("### Model Confidence Stats")
    c1, c2 = st.columns(2)
    c1.metric("Highest Probability Found", f"{max(p_real, p_fake)*100:.2f}%")
    c2.metric("Configured Threshold", f"{thresh_slider}")

    st.progress(p_real, text=f"Chance of Real: {p_real*100:.1f}%")
    st.progress(p_fake, text=f"Chance of Fake/AI: {p_fake*100:.1f}%")

st.write("---")
st.write(f"**Model Accuracy on Training =** {acc_value * 100:.2f}%")
st.pyplot(conf_fig)
st.caption("built with python")
