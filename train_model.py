import pandas as pd
import os, re, pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

curr_path = os.path.dirname(os.path.abspath(__file__))


def cleanText(s):
    s1 = str(s).lower()
    wrds = s1.split()
    ok = []
    for w in wrds:
        if "http" not in w and "www." not in w:
            ok.append(w)
    s2 = " ".join(ok)
    s3 = re.sub(r'[^a-z ]', ' ', s2)
    while "  " in s3:
        s3 = s3.replace("  ", " ")
    return s3.strip()


print("data padh raha...")

fakedf = pd.read_csv(os.path.join(curr_path, 'v_spam_db.csv'))
realdf = pd.read_csv(os.path.join(curr_path, 'v_legit_db.csv'))

fakedf['label'] = 1
realdf['label'] = 0

alldf = pd.concat([fakedf, realdf], ignore_index=True)
alldf = alldf.dropna(subset=['title', 'text', 'label'])

# shuffle nahi kiya tha pehle, model sirf ek side seekh raha tha
shuf = alldf.sample(frac=1, random_state=42).reset_index(drop=True)
shuf['combined'] = shuf['title'] + " " + shuf['text']

print("cleaning...")

corp = []
for t in shuf['combined']:
    corp.append(cleanText(t))
shuf['final'] = corp

# countvectorizer bhi try kiya tha, tfidf better tha
Vect = TfidfVectorizer(max_features=5000, stop_words="english")
Vect.fit(shuf['final'])
Xmat = Vect.transform(shuf['final'])
tgt = shuf['label']

Xtr, Xte, ytr, yte = train_test_split(Xmat, tgt, test_size=0.2, random_state=42)

print("training...")

# max_iter badhana pada, 100 pe ruk raha tha
mdl = LogisticRegression(max_iter=500)
mdl.fit(Xtr, ytr)

preds = mdl.predict(Xte)
acc = accuracy_score(yte, preds)
print(f"accuracy: {acc * 100:.2f}%")

o1 = open(os.path.join(curr_path, 'model.pkl'), 'wb')
pickle.dump(mdl, o1)
o1.close()

o2 = open(os.path.join(curr_path, 'vectorizer.pkl'), 'wb')
pickle.dump(Vect, o2)
o2.close()

o3 = open(os.path.join(curr_path, 'accuracy.txt'), 'w')
o3.write(str(acc))
o3.close()

print("sab save ho gaya")
