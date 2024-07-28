import os
import sys
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import clean_text
from src.utils import remove_stopwords



@dataclass
class data_preprocessing_config:
    IMDB_movie_data_preprocessed_path = os.path.join(os.getcwd(),'artifacts','IMDB_movie_data_preprocessed.csv')

class data_preprocessing:
    def __init__(self):
        self.data_preprocessing_config = data_preprocessing_config()
    
    def preprocesssing_level_1(self):
        try:
           # Reading data
           data_path = os.path.join(os.getcwd(),'data','IMDB_Movie_Review_Data.csv')
           data = pd.read_csv(data_path)
           logging.info(f"Data has beed accesed form the path\n{data_path}\n")
           logging.info(f"The dimesion of the dataset is:\n{data.shape}\n")
           logging.info(f"The dataset has {data.shape[0]} rows of data and {data.shape[1]} columns")
           
           # Handling missing values
           if data.isna().sum().values.sum() > 0:
               logging.info(f"Dataset consist of missing values\n{data.isna().sum()}\n")
               logging.info("Removing missing values")
               data.dropna(inplace=True)
               data.reset_index(drop=True,inplace=True)
               logging.info("Droped missing values")
               logging.info(f"The dimesion of the dataset after removing missing values is:\n{data.shape}\n")
           else:
               logging.info("There is no missing value in data")

           # Handling duplicate values
           if len(data[data.duplicated()]) > 0:
               logging.info(f"Data consist of {len(data[data.duplicated()])} duplicate values")
               logging.info("Handling duplicated values")
               data.drop_duplicates(keep='first',inplace=True)
               data.reset_index(drop=True,inplace=True)
               logging.info(f"The dimesion of the dataset after handling duplicated values is:\n{data.shape}\n")
           else:
               logging.info("There is no duplicated values")
            
           # Encoding sentiment column
           logging.info(f"Encoding sentiment column")
           data['sentiment'] = [0 if sentiment=='negative' else 1 for sentiment in data['sentiment']]
           logging.info(f"Encoding completed\n{data.head(10)}\n")
            
        except Exception as e:
            logging.info("Exception occured in preprocessing_level_1")
            raise CustomException(e,sys)
        
        return data
    
    
    def preprocesssing_level_2(self,level_1_preprocessed_data):
        try:
            nltk.download('stopwords')
            nltk.download('punkt')

            # Cleaning data
            data = level_1_preprocessed_data
            logging.info("Clenning the data by removing 'HTML tags', 'special characters and digits' and Convert to lowercase")
            logging.info(f"Example before cleanning:\n{data['review'][0]}\n")
            data['review'] = data['review'].apply(clean_text)
            logging.info(f"Example after cleanning:\n{data['review'][0]}\n")
            
            # Removing stopwords
            logging.info("Removing stopwords")
            data['review'] = data['review'].apply(word_tokenize)
            stop_words = set(stopwords.words('english'))
            logging.info(f"Example before removing stopwords:\n{data['review'][0]}\n")
            data['review'] = data['review'].apply(remove_stopwords)
            data['review'] = [' '.join(review) for review in data['review']]
            logging.info(f"Example after removing stopwords:\n{data['review'][0]}\n")
            
            data.to_csv(self.data_preprocessing_config.IMDB_movie_data_preprocessed_path,index=False,header=True)
            logging.info(f"Saved the preprocessed data at:\n{self.data_preprocessing_config.IMDB_movie_data_preprocessed_path}\n")

        except Exception as e:
            logging.info("Exception occured in preprocessing_level_2")
            raise CustomException(e,sys)
        
        
        return data,self.data_preprocessing_config.IMDB_movie_data_preprocessed_path
        