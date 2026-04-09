import pandas as pd
import os
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# yeh line isliye - relative path se run karo toh bhi sahi kaam kare
curr_path = os.path.dirname(os.path.abspath(__file__))


# yeh same function hai jo app.py mein bhi hai
# alag file mein duplicate karna pada kyunki train aur app dono ko chahiye tha
def do_text_cleaning(messy_string):
    step_one = str(messy_string).lower()
    tmp_words = step_one.split()
    ok_words = []
    
    # links hata do - yeh model ki accuracy gira rahe the
    for w in tmp_words:
        if "http" not in w and "www." not in w:
            ok_words.append(w)
            
    stage_two = " ".join(ok_words)
    
    # bas letters aur spaces chahiye, baaki sab hatao
    stage_three = re.sub(r'[^a-z ]', ' ', stage_two)
    
    # yeh loop isliye kyunki remove karne ke baad double spaces reh jaate hain
    while "  " in stage_three:
        stage_three = stage_three.replace("  ", " ")
        
    return stage_three.strip()


# --- data padhna ---

print("CSV files padh raha hoon...")

s_path = os.path.join(curr_path, 'v_spam_db.csv')
r_path = os.path.join(curr_path, 'v_legit_db.csv')

df_f = pd.read_csv(s_path)
df_r = pd.read_csv(r_path)

# 1 matlab fake, 0 matlab real - yeh manually set karna pada
df_f['label'] = 1
df_r['label'] = 0

all_data = pd.concat([df_f, df_r], ignore_index=True)

# kuch rows mein title ya text missing tha, drop kar diya
all_data = all_data.dropna(subset=['title', 'text', 'label'])

# shuffle karna zaroori tha warna pehle saare fake phir saare real aa rahe the
shuffled_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

# title aur text dono milake diye - sirf text se accuracy thodi kam thi
shuffled_data['temp_combined'] = shuffled_data['title'] + " " + shuffled_data['text']


# --- text clean karo ---

print("Text clean ho raha hai... thoda time lagega")

corpus_list = []
for txt in shuffled_data['temp_combined']:
    corpus_list.append(do_text_cleaning(txt))
    
shuffled_data['final_text'] = corpus_list


# --- vectorize karo ---

# pehle CountVectorizer try kiya tha, TfidfVectorizer better results aaye
vec_obj = TfidfVectorizer(max_features=5000, stop_words="english")
vec_obj.fit(shuffled_data['final_text'])
x_matrix = vec_obj.transform(shuffled_data['final_text'])

target_var = shuffled_data['label']

# standard 80/20 split, random_state fix kiya taki results same rahe baar baar
X_train, X_test, y_train, y_test = train_test_split(
    x_matrix, target_var, test_size=0.2, random_state=42
)


# --- model train karo ---

print("Training chal rahi hai...")

# logistic regression try kiya - simple hai lekin accuracy kaafi achhi aayi
# max_iter badhana pada kyunki default 100 pe converge nahi ho raha tha
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)

my_preds = log_model.predict(X_test)
curr_acc = accuracy_score(y_test, my_preds)
print(f"Ho gaya! Accuracy: {curr_acc * 100:.2f}%")


# --- save karo ---

# model pkl mein save karo taki app.py ko baar baar train na karna pade
out1 = open(os.path.join(curr_path, 'model.pkl'), 'wb')
pickle.dump(log_model, out1)
out1.close()

# vectorizer alag save kiya - yeh bhi chahiye hoga prediction ke time
out2 = open(os.path.join(curr_path, 'vectorizer.pkl'), 'wb')
pickle.dump(vec_obj, out2)
out2.close()

# accuracy txt mein save ki taki app mein show ho sake
out3 = open(os.path.join(curr_path, 'accuracy.txt'), 'w')
out3.write(str(curr_acc))
out3.close()

print("model.pkl, vectorizer.pkl, accuracy.txt - sab save ho gaya")
