from src.components.data_preprocessing import data_preprocessing
from src.components.word_embedding import word_embedding
from src.pipelines.model_training import model_traning_saving

if __name__ == "__main__":
    preprocessor = data_preprocessing()
    embedder = word_embedding()
    model_trainer = model_traning_saving()

    data = preprocessor.preprocesssing_level_1()
    data,preprocessed_data_path = preprocessor.preprocesssing_level_2(data)
    data,embedded_data_path = embedder.embedding(preprocessed_data_path)
    X_train,X_test,y_train,y_test = model_trainer.get_data_for_model_building(embedded_data_path)
    model_path = model_trainer.model_building(X_train,y_train)
    model_trainer.model_accuracy_check(model_path,X_test,y_test)
    print("Done")
