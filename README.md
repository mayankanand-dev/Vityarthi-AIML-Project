# Vityarthi AIML Project: AI Fake Article Detector

## About the Project
Hey! So this is my submission for our AI/ML project. Basically, I wanted to build something that could figure out if a news article or paragraph is real or if it’s fake/AI-generated. There’s so much AI text floating around these days, so I thought it would be cool to see if I could catch it using a basic machine learning model. It’s definitely not perfect, but it works surprisingly well for most things I threw at it!

## Features
* Simple web UI where you just paste your text and hit check
* Shows a probability bar of how likely the text is real vs fake
* Has a slider to manually adjust the strictness/threshold (because sometimes the model is a bit too sensitive)
* Shows a confusion matrix at the bottom so you can see how the model actually performed during training
* Pretty fast since it trains on startup and just caches the model

## Tech Stack
* **Python** (main language)
* **Streamlit** (for the web frontend, literally a lifesaver for making UI fast)
* **Scikit-learn** (used Logistic Regression and TF-IDF for the text stuff)
* **Pandas** (handling the csv datasets)
* **Matplotlib & Seaborn** (just for plotting the confusion matrix graph)

## How It Works
Honestly, it's pretty straightforward. I have two massive CSV files of spam and legit articles. When you first run the app, it loads them up, drops missing values, and cleans the text (I had to write a custom cleaner because `apply()` in pandas was acting weird lol). It uses TF-IDF to turn the words into numbers/vectors, and then a standard Logistic Regression model learns the patterns. When you paste new text, it just runs it through that same pipeline and spits out the probability.

## Screenshots
Here's what the app looks like when it's starting up:
![Loading Screen](screenshots/loading_page.png)

The main interface before you paste anything:
![Landing Page](screenshots/landing_page.png)

When the model catches some fake/AI text:
![Fake Text Detected](screenshots/fake_detected.png)

When the text seems legit:
![Real Text Detected](screenshots/real_detected.png)

## Installation
If you actually want to run this yourself on your laptop, here's how:

1. Clone this repo or download the files.
2. Make sure you have python installed.
3. Install the packages from the requirements file. I just did:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the streamlit app:
   ```bash
   streamlit run app.py
   ```
5. It might take a minute the very first time because it has to read the csv files and train the model. Just let it spin.

## Usage
Once the browser tab opens, just grab a paragraph of text from somewhere (a news site, ChatGPT, etc.) and paste it into the big text box. Hit "Check Article" and wait a second. You can mess with the "Strictness level" slider if it's giving you false positives. I found that leaving it at around 0.7 works best for my testing.

## Notes / Limitations
* The datasets are kinda old, so if you paste brand new news it might get confused.
* It struggles with super short text. I added a 5-word minimum check, but honestly, it needs at least a proper paragraph to make a good guess.
* Because I'm training the model every time the app starts, it eats up a bit of RAM. I tried saving the `.pkl` files but it was getting messy, so I just used Streamlit's `@st.cache_resource` to cache it after the first load. 

## Future Improvements
If I had more time before the deadline, I'd probably:
* Pre-train the model and save the weights so startup is instant
* Try out a Naive Bayes model to see if it's better than Logistic Regression
* Clean up the UI a bit more, maybe add a dark mode
