import pandas as pd
import os
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

curr_path = os.path.dirname(os.path.abspath(__file__))

def do_text_cleaning(messy_string):
    step_one = str(messy_string).lower()
    tmp_words = step_one.split()
    ok_words = []
    for w in tmp_words:
        if "http" not in w and "www." not in w:
            ok_words.append(w)
    stage_two = " ".join(ok_words)
    stage_three = re.sub(r'[^a-z ]', ' ', stage_two)
    while "  " in stage_three:
        stage_three = stage_three.replace("  ", " ")
    return stage_three.strip()

print("Reading CSV files...")
df_f = pd.read_csv(os.path.join(curr_path, 'v_spam_db.csv'))
df_r = pd.read_csv(os.path.join(curr_path, 'v_legit_db.csv'))

df_f['label'] = 1
df_r['label'] = 0

all_data = pd.concat([df_f, df_r], ignore_index=True)
all_data = all_data.dropna(subset=['title', 'text', 'label'])
shuffled_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_data['temp_combined'] = shuffled_data['title'] + " " + shuffled_data['text']

print("Cleaning text...")
corpus_list = [do_text_cleaning(txt) for txt in shuffled_data['temp_combined']]
shuffled_data['final_text'] = corpus_list

print("Training model...")
vec_obj = TfidfVectorizer(max_features=5000, stop_words="english")
vec_obj.fit(shuffled_data['final_text'])
x_matrix = vec_obj.transform(shuffled_data['final_text'])
target_var = shuffled_data['label']

X_train, X_test, y_train, y_test = train_test_split(
    x_matrix, target_var, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)

my_preds = log_model.predict(X_test)
curr_acc = accuracy_score(y_test, my_preds)
print(f"Model accuracy: {curr_acc * 100:.2f}%")

# save everything
with open(os.path.join(curr_path, 'model.pkl'), 'wb') as f:
    pickle.dump(log_model, f)

with open(os.path.join(curr_path, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vec_obj, f)

with open(os.path.join(curr_path, 'accuracy.txt'), 'w') as f:
    f.write(str(curr_acc))

print("Saved model.pkl, vectorizer.pkl, accuracy.txt")
print("You can now remove the CSV files from git.")
