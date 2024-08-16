import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            model = load_object(file_path=model_path)

            preprocessor_path = 'artifacts/preprocessor.pkl'
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(
            self,
            age: int,
            gender: str,
            tumor_type: str,
            tumor_grade: str,
            tumor_location: str, 
            treatment: str,
            treatment_outcome: str,
            recurrence_time: int,
            recurrence_site: str):
        
        self.age = age
        self.gender = gender
        self.tumor_type = tumor_type
        self.tumor_grade = tumor_grade
        self.tumor_location = tumor_location
        self.treatment = treatment
        self.treatment_outcome = treatment_outcome
        self.recurrence_time = recurrence_time
        self.recurrence_site = recurrence_site

    
    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                'Age': [self.age],
                'Gender': [self.gender],
                'Tumor Type': [self.tumor_type],
                'Tumor Grade': [self.tumor_grade],
                'Tumor Location': [self.tumor_location],
                'Treatment': [self.treatment],
                'Treatment Outcome': [self.treatment_outcome],
                'Time to Recurrence (months)': [self.recurrence_time],
                'Recurrence Site': [self.recurrence_site]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)