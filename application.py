import re
import os
import time
import warnings
import numpy as np
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.utils import padding, clean_text, remove_stopwords, one_hot_encoding
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('punkt')

model_path = os.path.join(os.getcwd(), 'models', 'movie_review_sentiment_analyzer_simple_rnn.h5')
model = load_model(model_path)

def create_word_cloud(cleaned_tokens):
    input_text = ' '.join(cleaned_tokens)
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='white').generate(input_text)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)  

def vector_tranformation(cleaned_tokens):
    input_text = ' '.join(cleaned_tokens)
    one_hot_representation = one_hot_encoding(input_text)
    padded_one_hot_representation = padding(one_hot_representation)
    X = np.array(list(padded_one_hot_representation))
    return X

def preprocessing(text):
    cleaned_text = clean_text(text)
    tokenized_words = word_tokenize(cleaned_text)
    cleaned_tokens = remove_stopwords(tokenized_words)
    create_word_cloud(cleaned_tokens)
    X = vector_tranformation(cleaned_tokens)
    return X


st.markdown(f"<h1 style='text-align: center; font-size: 48px; color: blue'>IMDBMoodAnalyzer: RNN-Powered Sentiment Analysis of IMDB Movie Reviews</h1>", unsafe_allow_html=True)


st.write("Write a movie review")

with st.sidebar:
    
    st.markdown(f"<h2 style='text-align: center; font-size: 24px;'>Try a sample review:</h2>", unsafe_allow_html=True)
    reviews = [
        "I loved this movie! The acting was superb and the plot was engaging.",
        "watched film really expecting much got pack films pretty terrible way fiver could expect know right terrible movie stretching interesting points occasional camcorder view nice touch drummer like drummer ie damned annoying well thats actually problem boring assume attempt build tension whole lot nothing happens utterly tedious thumb fast forward button ready press movie gave go seriously lead singer band great looking coz dont half mention beautiful hell lot thought looked bit like meercat havent even mentioned killer im even gon na go worth explaining anyway far im concerned star london reason watch exception london actually quite funny wasnt acting talent ive certainly seen lot worse ive also seen lot better best avoid unless bored watching paint dry",
        "one reviewers mentioned watching oz episode youll hooked right exactly happened methe first thing struck oz brutality unflinching scenes violence set right word go trust show faint hearted timid show pulls punches regards drugs sex violence hardcore classic use wordit called oz nickname given oswald maximum security state penitentary focuses mainly emerald city experimental section prison cells glass fronts face inwards privacy high agenda em city home manyaryans muslims gangstas latinos christians italians irish moreso scuffles death stares dodgy dealings shady agreements never far awayi would say main appeal show due fact goes shows wouldnt dare forget pretty pictures painted mainstream audiences forget charm forget romanceoz doesnt mess around",
        "films first shot keira knightley elizabeth bennet wandering reading field dawn thus invoking clichs cinema developed address phenomenon strongminded rebellious female character period drama knew something make want kill myselfjoe wright seemed read book regrettable misapprehension filming fact jane austens subtle nuanced comedy manners conducted sparkling delicate social interaction eighteenth century english drawingrooms sort ucertificate wuthering heights thus treated every scene elizabeth darcy taking place outside apparent reason inappropriately rugged scenery often pouring rain mention jane austen particular p p passion sexual tension love different strategies negotiating stultification eighteenth century society completely ignored bennets house rambunctious chaotic place everybody shouts runs around leaves underwear chairs pigs wander happily house society balls become rowdy country dances one step away matrix reloaded style danceorgy everybody says exactly think without slightest regard proprietythe genius jane austen lies exploring void created society nobody says think mean overwhelming regard propriety tragic predicaments characters arise misunderstandings miscommunications enabled speechless gap brilliance jane austen factor allows plots particularly film function completely erased subtlety general nowhere int film sacrificed favour overwrought drama jarred entirely material performancesit obviously trying serious film humour pride prejudice austens methodology appeal almost entirely suppressed favour pofaced melodrama allowed handled clumsily pride prejudice serious narrative makes serious points yes serious points weightier themes intertwined humour embedded cant lose jane austens technique leaving bare bones story expect themes remain even replace techniques heavyhanded mysticalnuminous fauxbrow cinematographyelizabeth bennett supposed woman adult mature sensible clearsighted keira knightley played first half film like emptyheaded giggling schoolgirl second half like emptyheaded schoolgirl thinks tragic heroine elizabeths wit combative verbal exchanges quintessential characteristic able see laugh everybodys follies including strength composure fantastic clearsightedness completely lost replaced lot giggling staring distance rather able keep head losing started cry scream slightest provocation genuinely raging either petulant hissy fits great strength austens elizabeth least austens eyes ability retain integrity observance remaining within boundaries society sustaining impeachable propriety knightleys elizabeth regard whatsoever convention furthermore seemed think wandering around barefoot mud eighteenth century version overalls established beyond doubt spirited strongminded therefore nothing character written performance sustain astonishingly unsubtle bland performance quest blandness weakness ably matched matthew macfaydendonald sutherland mr bennet seemed weak ineffectual permanently befuddled without wicked sense humour ironic detachment expense human relationships makes mr bennet fascinating tragic special bond lizzie two sensible people world fools completely lost least fools world fools completely deprived end film emotional impact mr bingley longer amiable wellmeaning point folly played complete retard cheap laughs woman playing jane wildly inconsistent may well tried anything character script veered wildly verbatim chunks jane austen delivered remarkable clumsiness totally contemporaneous language would place modern day romantic comedyjust get bbc adaptation dvd save heartache"

    ]
    selected_review = st.selectbox("Select a review", reviews)


user_input = st.text_area('', value=selected_review if selected_review else '', height=200)


if st.button("Get Result"):
    with st.spinner("Sentiment analysis is being done, based on your review size it will take time... wait!"):
        if user_input.strip():
            X = preprocessing(user_input)
            prediction = model.predict(X)
    
            if prediction[0][0] >= 0.5:
                sentiment = "Positive"
            else:
                sentiment = "Negative"
    
            if sentiment == "Positive":
                st.markdown(f"<p style='text-align: center; font-size: 34px; color: green'>{sentiment}</p>", unsafe_allow_html=True)
            elif sentiment == "Negative":
                st.markdown(f"<p style='text-align: center; font-size: 34px; color: red'>{sentiment}</p>", unsafe_allow_html=True)
        else:
           st.write("Please enter a movie review.")


st.markdown(f"<p style='text-align: center; font-size: 18px;'>Check out the code on <a href='https://github.com/Aman-Vishwakarma1729/IMDBMoodAnalyzerr'>GitHub</a></p>", unsafe_allow_html=True)


st.markdown(f"<p style='text-align: center; font-size: 12px;'>Copyright 2024 Aman Vishwakarma. All rights reserved.</p>", unsafe_allow_html=True)