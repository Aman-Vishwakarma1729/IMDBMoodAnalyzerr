# <div align="center">IMDBMoodAnalyzer: RNN-Powered Sentiment Analysis of IMDB Movie Reviews</div>
<div align="center">
  <img src="readme_data\output.png" alt="Designer" width="500"/>
</div>

## Table of content
--------------
1. [Introduction](#introduction)
2. [About Dataset](#about-data)
3. [Tools and Techniques Used](#tools-and-techniques-used)
4. [How to use](#how-to-use)

## Introduction
--------------
IMDBMoodAnalyzer is an advanced deeplearning project designed using Natural Language Pricessing to perform sentiment analysis on movie reviews from the IMDB dataset. Utilizing a Recurrent Neural Network (RNN) architecture, this project aims to accurately classify the sentiment of movie reviews as either positive or negative, providing valuable insights into audience perceptions. Leveraging the power of Recurrent Neural Networks to capture temporal dependencies and contextual nuances in text data. It includes extensive text preprocessing steps such as tokenization, stopwords removal, and word embeddings to ensure high-quality input data. it generates word clouds from cleaned review texts to visualize the most frequent words and themes. Built with Streamlit, providing a user-friendly interface for inputting movie reviews and obtaining real-time sentiment predictions. This project can easily be adapted for sentiment analysis of customer reviews in various domains such as e-commerce, hospitality, and more.

## About Dataset
----------------
* The data source [IMDB Movie Review dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* The IMDB movie review dataset is a large-scale dataset containing movie reviews and corresponding sentiment labels.
* The dataset is derived from the Internet Movie Database (IMDB), a comprehensive online database of film information.
* he dataset comprises 50,000 movie reviews.

## Tools and Techniques Used
----------------------------
* Python
* Anaconda (To create virtual environment)
* Streamlit (For Deployment)
* RNN (Recurrent Neural Network)
* Embedding (To convert words to vector)
* Tensoflow

## How to use
-------------
- clone this repo
- unzip the data in "data" folder and get .csv file
- pip install -r requirements.txt
- python main.py
- streamlit run application.py
* To understand the entire workflow go to folder named 'research_notebooks' and go through the notebook in it.
