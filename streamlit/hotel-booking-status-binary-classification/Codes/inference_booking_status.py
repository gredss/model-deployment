
import pickle
import warnings
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def load_encoders(encoder_files):
    encoders = {}
    for col, filename in encoder_files.items():
        with open(filename, 'rb') as file:
            encoders[col] = pickle.load(file)
    return encoders

def preprocess_user_input(user_input, encoders):
    user_input_df = pd.DataFrame([user_input])

    for col, encoder in encoders.items():
        if col in user_input_df.columns:
            encoded_input = encoder.transform(user_input_df[[col]]).toarray()
            encoded_columns = encoder.get_feature_names_out()
            encoded_df = pd.DataFrame(encoded_input, columns=encoded_columns)
            user_input_df = pd.concat([user_input_df, encoded_df], axis=1).drop(columns=[col])

    return user_input_df

def predict_with_model(model, user_input_df):
    prediction = model.predict(user_input_df)
    return prediction[0]

def main():
    model_filename = 'XGB_trained_model.pkl'

    encoder_files = {
        'type_of_meal_plan': 'oneHot_encode_meal.pkl',
        'room_type_reserved': 'oneHot_encode_room.pkl',
        'market_segment_type': 'oneHot_encode_market.pkl',
    }
    
    model = load_model(model_filename)
    encoders = load_encoders(encoder_files)

    user_input = {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 1,
        'no_of_week_nights': 2,
        'required_car_parking_space': 1,
        'lead_time': 30,
        'arrival_year': 2025,
        'arrival_month': 4,
        'arrival_date': 15,
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 2,
        'avg_price_per_room': 200,
        'no_of_special_requests': 1,
        'type_of_meal_plan_Meal Plan 1': 1,
        'type_of_meal_plan_Meal Plan 2': 0,
        'type_of_meal_plan_Meal Plan 3': 0,
        'type_of_meal_plan_Not Selected': 0,
        'room_type_reserved_Room_Type 1': 0,
        'room_type_reserved_Room_Type 2': 1,
        'room_type_reserved_Room_Type 3': 0,
        'room_type_reserved_Room_Type 4': 0,
        'room_type_reserved_Room_Type 5': 0,
        'room_type_reserved_Room_Type 6': 0,
        'room_type_reserved_Room_Type 7': 0,
        'market_segment_type_Aviation': 0,
        'market_segment_type_Complementary': 1,
        'market_segment_type_Corporate': 0,
        'market_segment_type_Offline': 0,
        'market_segment_type_Online': 0
    }

    user_input_df = preprocess_user_input(user_input, encoders)
    prediction = predict_with_model(model, user_input_df)
    print(f"The predicted booking status is: {'Canceled' if prediction == 1 else 'Not Canceled'}")

if __name__ == "__main__":
    main()
