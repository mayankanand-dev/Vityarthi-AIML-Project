import pandas as pd
import os
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# figuring out where this file is so paths work everywhere
curr_path = os.path.dirname(os.path.abspath(__file__))

# same cleaning function as in app.py, copy pasted it here
def do_text_cleaning(messy_string):
    step_one = str(messy_string).lower()
    tmp_words = step_one.split()
    ok_words = []
    
    # removing links because they were messing up the model
    for w in tmp_words:
        if "http" not in w and "www." not in w:
            ok_words.append(w)
            
    stage_two = " ".join(ok_words)
    
    # strip out anything thats not a letter
    stage_three = re.sub(r'[^a-z ]', ' ', stage_two)
    
    # double spaces kept showing up after removing stuff
    while "  " in stage_three:
        stage_three = stage_three.replace("  ", " ")
        
    return stage_three.strip()

# ---- reading the data ----
print("Reading CSV files...")

s_path = os.path.join(curr_path, 'v_spam_db.csv')
r_path = os.path.join(curr_path, 'v_legit_db.csv')

df_f = pd.read_csv(s_path)
df_r = pd.read_csv(r_path)

# 1 = fake, 0 = real
df_f['label'] = 1
df_r['label'] = 0

all_data = pd.concat([df_f, df_r], ignore_index=True)

# dropping rows with missing values
all_data = all_data.dropna(subset=['title', 'text', 'label'])

# shuffle so fake and real arent grouped together
shuffled_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

# combining title and article text into one column for better accuracy
shuffled_data['temp_combined'] = shuffled_data['title'] + " " + shuffled_data['text']

# ---- cleaning ----
print("Cleaning text... (takes a while)")

corpus_list = []
for txt in shuffled_data['temp_combined']:
    corpus_list.append(do_text_cleaning(txt))
    
shuffled_data['final_text'] = corpus_list

# ---- vectorizing ----
# tfidf worked way better than countvectorizer in my testing
vec_obj = TfidfVectorizer(max_features=5000, stop_words="english")
vec_obj.fit(shuffled_data['final_text'])
x_matrix = vec_obj.transform(shuffled_data['final_text'])

target_var = shuffled_data['label']

# standard 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    x_matrix, target_var, test_size=0.2, random_state=42
)

# ---- training ----
print("Training model...")

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)

my_preds = log_model.predict(X_test)
curr_acc = accuracy_score(y_test, my_preds)
print(f"Done! Accuracy came out to: {curr_acc * 100:.2f}%")

# ---- saving ----
# saving model
out1 = open(os.path.join(curr_path, 'model.pkl'), 'wb')
pickle.dump(log_model, out1)
out1.close()

# saving vectorizer separately so i can load it in app.py
out2 = open(os.path.join(curr_path, 'vectorizer.pkl'), 'wb')
pickle.dump(vec_obj, out2)
out2.close()

# also saving accuracy so the app can display it without retraining
out3 = open(os.path.join(curr_path, 'accuracy.txt'), 'w')
out3.write(str(curr_acc))
out3.close()

print("Saved model.pkl, vectorizer.pkl, accuracy.txt - all done")
