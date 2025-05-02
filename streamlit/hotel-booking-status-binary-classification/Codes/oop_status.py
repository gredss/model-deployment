
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        if 'Booking_ID' in self.df.columns:
            self.df.drop(columns=['Booking_ID'], inplace=True)

    def split_input_output(self, target_column='booking_status'):
        self.input_df = self.df.drop(columns=[target_column])
        self.output_df = self.df[target_column]


class Preprocessor:
    def __init__(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test
        self.categorical = []
        self.numerical_binary = []
        self.numerical_nonbinary = []

    def categorize_columns(self):
        for col in self.x_train.columns:
            if self.x_train[col].dtype == "object":
                self.categorical.append(col)
            else:
                unique_vals = self.x_train[col].dropna().unique()
                if len(unique_vals) == 2:
                    self.numerical_binary.append(col)
                elif len(unique_vals) > 2:
                    self.numerical_nonbinary.append(col)

    def handle_missing_values(self):
        self.x_train['type_of_meal_plan'].fillna('Not Selected', inplace=True)
        self.x_test['type_of_meal_plan'].fillna('Not Selected', inplace=True)
        self.x_train['required_car_parking_space'].fillna(0.0, inplace=True)
        self.x_test['required_car_parking_space'].fillna(0.0, inplace=True)
        median_price = self.x_train['avg_price_per_room'].median()
        self.x_train['avg_price_per_room'].fillna(median_price, inplace=True)
        self.x_test['avg_price_per_room'].fillna(median_price, inplace=True)


class EncoderHandler:
    def __init__(self, x_train, x_test, categorical_columns):
        self.x_train = x_train
        self.x_test = x_test
        self.categorical_columns = categorical_columns

    def fit_transform(self):
        for col in self.categorical_columns:
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoded_train = pd.DataFrame(
                encoder.fit_transform(self.x_train[[col]]).toarray(),
                columns=encoder.get_feature_names_out()
            )
            encoded_test = pd.DataFrame(
                encoder.transform(self.x_test[[col]]).toarray(),
                columns=encoder.get_feature_names_out()
            )

            self.x_train = pd.concat([self.x_train.reset_index(drop=True), encoded_train], axis=1)
            self.x_test = pd.concat([self.x_test.reset_index(drop=True), encoded_test], axis=1)

        drop_cols = [col for col in self.categorical_columns if col in self.x_train.columns]
        self.x_train.drop(columns=drop_cols, inplace=True)
        self.x_test.drop(columns=drop_cols, inplace=True)


class ModelHandler:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def encode_target(self):
        self.y_train = self.y_train.replace({'Canceled': 1, 'Not_Canceled': 0})
        self.y_test = self.y_test.replace({'Canceled': 1, 'Not_Canceled': 0})
        y_encode = {'Canceled': 1, 'Not_Canceled': 0}
        with open('y_encode.pkl', 'wb') as f:
            pickle.dump(y_encode, f)

    def train_model(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=4)
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.x_test)
        print('\nClassification Report\n')
        print(classification_report(self.y_test, y_pred, target_names=['Not_Canceled', 'Canceled']))

    def save_model(self, filename='XGB_trained_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f'Model saved as {filename}')


if __name__ == '__main__':
    file_path = 'Dataset_B_hotel.csv'

    data_handler = DataHandler(file_path)
    data_handler.load_data()
    data_handler.split_input_output()

    x_train, x_test, y_train, y_test = train_test_split(
        data_handler.input_df, data_handler.output_df, test_size=0.2, random_state=42)

    preprocessor = Preprocessor(x_train, x_test)
    preprocessor.categorize_columns()
    preprocessor.handle_missing_values()

    encoder_handler = EncoderHandler(preprocessor.x_train, preprocessor.x_test, preprocessor.categorical)
    encoder_handler.fit_transform()

    model_handler = ModelHandler(encoder_handler.x_train, encoder_handler.x_test, y_train, y_test)
    model_handler.encode_target()
    model_handler.train_model()
    model_handler.evaluate_model()
    model_handler.save_model()
