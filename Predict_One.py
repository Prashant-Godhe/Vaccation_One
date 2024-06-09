import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('HOTEL_TEST_DATA - Sheet1.csv')

# Display the first few rows of the dataframe
print(df.head())

# Assume the CSV has columns: 'city', 'price', 'hotel'
# If there are more columns, select only the relevant ones
df = df[['city', 'price', 'property_name']]
# Encode categorical variables
city_encoder = LabelEncoder()
hotel_encoder = LabelEncoder()

# Encode categorical variables
df['city_encoded'] = city_encoder.fit_transform(df['city'])
df['hotel_encoded'] = hotel_encoder.fit_transform(df['property_name'])

# Features and target variable
X = df[['city_encoded', 'price']]
y = df['hotel_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Function to predict the hotel based on user input
def predict_hotel(city, price):
    try:
        city_encoded = city_encoder.transform([city])[0]
    except ValueError:
        return "City not recognized by the model."

    features = [[city_encoded, price]]
    hotel_encoded = model.predict(features)[0]
    hotel = hotel_encoder.inverse_transform([hotel_encoded])[0]
    return hotel

# Example usage

city = input("Enter city name: ")

price = int(input("Enter your price: "))
city_preference = 'Goa'
price_preference = 1100

predicted_hotel = predict_hotel(city, price)
print(f'Predicted Hotel: {predicted_hotel}')