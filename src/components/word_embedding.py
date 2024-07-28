import os
import sys
import re
import pandas as pd
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing import sequence
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import one_hot_encoding
from src.utils import padding


@dataclass
class data_embedding_config:
    Embedded_IMDB_movie_data_preprocessed_path = os.path.join(os.getcwd(),'artifacts','Embedded_IMDB_movie_data_preprocessed.csv')

class word_embedding:
    def __init__(self):
        self.data_embedding_config = data_embedding_config()
    
    def embedding(self,preprocessed_data_path):
        try:
            # Reading dataset
            logging.info("Word embedding has started")
            data = pd.read_csv(preprocessed_data_path)
            review_list = data['review'].values
            
            # One-Hot Encoding
            logging.info("One Hot Encoding")
            one_hot_representation = one_hot_encoding(review_list)
            padded_one_hot_representation = padding(one_hot_representation)
            data['review'] = list(padded_one_hot_representation)
            logging.info(f"After embedding the dataset looks like:\n{data.head(10)}\n")
            
            data.to_csv(self.data_embedding_config.Embedded_IMDB_movie_data_preprocessed_path,index=False,header=True)
            logging.info(f"Saved the preprocessed data at:\n{self.data_embedding_config.Embedded_IMDB_movie_data_preprocessed_path}\n")

        except Exception as e:
            logging.info("Exception occured in word embedding")
            raise CustomException(e,sys)
        
        return data,self.data_embedding_config.Embedded_IMDB_movie_data_preprocessed_path