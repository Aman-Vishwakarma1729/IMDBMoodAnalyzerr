import os
import sys
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from dataclasses import dataclass

@dataclass
class model_training_config:
    model_path = os.path.join(os.getcwd(),'models','movie_review_sentiment_analyzer_simple_rnn.h5')

class model_traning_saving:
    def __init__(self):
        self.model_training_config = model_training_config()

    def get_data_for_model_building(self,embedded_data_path):
            try:
                # Reading dataset
                logging.info("Got the final data")
                data = pd.read_csv(embedded_data_path)
                
                # Converting data to numpy array
                X = np.array(data['review'].tolist())
                y = np.array(data['sentiment'].tolist())
                logging.info("Converted the data to numpy array")
                logging.info(f"Input data is:\n{X}\n")
                logging.info(f"Target data is:\n{y}\n")

                # Splitting dataset in training and testing
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)
                logging.info(f"Train input data dimension: {X_train.shape}\n")
                logging.info(f"Test input data dimension: {X_test.shape}\n")
                logging.info(f"Train target data dimension: {y_train.shape}\n")
                logging.info(f"Test target data dimension: {y_test.shape}\n")
                            
            except Exception as e:
                logging.info("Exception occured in geting data for model building")
                raise CustomException(e,sys)
            
            return X_train,X_test,y_train,y_test

    def model_building(self,X_train,y_train):
        try:
            logging.info("Model building started")
            voc_size = 5000
            max_len = 500
            model = Sequential()
            model.add(Embedding(voc_size,128,input_length=max_len))
            model.add(SimpleRNN(128,activation='relu'))
            model.add(Dense(1,activation='sigmoid'))

            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

            logging.info(f"Model summary:\n{model.summary()}\n")

            earlystopping = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

            history = model.fit(
                                    X_train,y_train,
                                    epochs=10,
                                    batch_size=32,
                                    validation_split=0.20,
                                    callbacks = [earlystopping]
                                )
            
            model_path = os.path.join(os.getcwd(),'models','movie_review_sentiment_analyzer_simple_rnn.h5')
            model.save(model_path)
            logging.info(f"The model is trained and saved at:\n{model_path}\n")

        except Exception as e:
                logging.info("Exception occured while model building")
                raise CustomException(e,sys)
            
        return model_path

    def model_accuracy_check(self,model_path,X_test,y_test):
        try:
            logging.info('Getting accuracy of the model based on test data')
            model =  load_model(model_path)
            loss, accuracy = model.evaluate(X_test, y_test)
            logging.info(f"ccuracy of model is: \n{accuracy * 100:.2f}%\n")

        except Exception as e:
                logging.info("Exception occured while cheking model accuracy")
                raise CustomException(e,sys)

