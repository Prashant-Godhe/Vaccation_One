#All Imports
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
import requests
# import seaborn as sns
# import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('goibibo_com-travel_sample_05-06-24 - goibibo_com-travel_sample.csv.csv')

places_df = pd.read_csv('Top Indian Places to Visit.csv')

food_df = pd.read_csv('swiggy.csv')

fares_df = pd.read_csv('Distance_Fares - Sheet1.csv')

fares_df = pd.DataFrame(fares_df)
# Display the first few rows of the dataframe
df.head()

# print("Missing values:\n", df.isnull().sum())

df_df = df[['city','property_name', 'address','hotel_category','point_of_interest','property_type','room_facilities','hotel_facilities', 'state', 'Price', 'guests_no', 'additional_info']]
#df_df.head()

#2. Use the most frequent value:
df_df['point_of_interest'] = df['point_of_interest'].fillna(df['point_of_interest'].mode()[0])
df_df['room_facilities'] = df['room_facilities'].fillna(df['room_facilities'].mode()[0])
df_df['hotel_facilities'] = df['hotel_facilities'].fillna(df['hotel_facilities'].mode()[0])
df_df['additional_info'] = df['additional_info'].fillna(df['additional_info']).mode()[0]

# Encode categorical variables
city_encoder = LabelEncoder()
hotel_encoder = LabelEncoder()
guest_encoder = LabelEncoder()

# Check unique values before encoding
# print("Unique cities before encoding:", df_df['city'].unique())
# print("Unique hotels before encoding:", df_df['property_name'].unique())

# Encode categorical variables
df_df['city_encoded'] = city_encoder.fit_transform(df_df['city'])
df_df['hotel_encoded'] = hotel_encoder.fit_transform(df_df['property_name'])
df_df['guest_encoded'] = guest_encoder.fit_transform(df_df['guests_no'])

# Features and target variable
X = df_df[['city_encoded', 'Price','guests_no']]
y = df_df['hotel_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the distribution in training and test sets
# print("Training set distribution:\n", y_train.value_counts())
# print("Test set distribution:\n", y_test.value_counts())

# Choose a machine learning model
model = RandomForestClassifier(n_estimators=10, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
#
# # Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# st.write(f'Accuracy: {accuracy * 100:.2f}%')
#
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
# st.write('Confusion Matrix:')
# st.pyplot(fig)
#
# # Classification Report
# report = classification_report(y_test, y_pred, target_names=hotel_encoder.classes_)
# st.write('Classification Report:')
# st.text(report)
#
# # Log Loss
# y_prob = model.predict_proba(X_test)
# loss = log_loss(y_test, y_prob)
# st.write(f'Log Loss: {loss}')


# Function to predict the hotel based on user input
def predict_hotel(city, price, guest):
    try:
        city_encoded = city_encoder.transform([city])[0]
    except ValueError:
        return "City not recognized by the model."

    features = [[city_encoded, price, guest]]
    hotel_encoded = model.predict(features)[0]
    hotel = hotel_encoder.inverse_transform([hotel_encoded])[0]
    hotel_city = df_df[df_df['property_name'] == hotel]['city'].values[0]
    if hotel_city != city:
      return "No hotel found in the specified city and price range."
    else:
        # Retrieve the address of the predicted hotel
        hotel_info = df_df[df_df['property_name'] == hotel].iloc[0]
        address = hotel_info['address']

        room_features = hotel_info['room_facilities']

        hotel_features = hotel_info['hotel_facilities']

        return {"hotel": hotel, "address": address, "room_features": room_features, "hotel_features":hotel_features}



def get_hotel_image(hotel_name):
    API_KEY = 'AIzaSyDqVOkllQJ_6cZsKY3nhZMN_FVFR_oIm5A'
    SEARCH_ENGINE_ID = 'f33251543fb274615'
    url = 'https://www.googleapis.com/customsearch/v1'

    search_queries = [
        f'{hotel_name} hotel exterior',
        f'{hotel_name} hotel room',
        f'{hotel_name} hotel amenities'
    ]
    images = []

    for query in search_queries:
        params = {
            'q': query,
            'key': API_KEY,
            'cx': SEARCH_ENGINE_ID,
            'searchType': 'image',
            'num': 5  # Number of images to retrieve per query
        }

        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        results = response.json().get('items', [])
        images.extend(results)

        if len(images) >= 5:  # If we have enough images, break the loop
            break

    return images[:5]  # Return only the top 5 images


def get_hotel_info(hotel_name):
  # Get the hotel information from the dataframe.
  hotel = df_df[df_df['property_name'] == hotel_name]

  if hotel.empty:
        # Handle the case where no hotel is found
        return {'address': 'Hotel not found'}
    # Return a dictionary with both the name and address.
  return {
    'address': hotel['address'].values[0]
  }


def get_places_to_visit(city):
    places = places_df[places_df['City'].str.lower() == city.lower()]
    if places.empty:
        return "No places to visit found for this city."
    return places['Name'].tolist()



st.title("Vacation Recommendation One")
st.text("Welcome \nThis is the one website to plan your vacation")

def get_food_spot(city):
    food_spots = food_df[food_df['City'].str.lower() == city.lower()]
    if food_spots.empty:
        return "No places to visit found for this city."
    return food_spots['Restaurant'].tolist()


def get_fares(city, ori_city):
    # Normalize input strings
    city = city.strip().lower()
    ori_city = ori_city.strip().lower()

    # Ensure the columns 'Destination' and 'Origin' exist in the DataFrame
    if 'Destination' not in fares_df.columns or 'Origin' not in fares_df.columns:
        raise KeyError("The DataFrame does not have the required 'Destination' or 'Origin' columns")

    # Filter the DataFrame for the rows matching the given cities
    matching_fares = fares_df[
        (fares_df['Destination'].str.lower() == city) &
        (fares_df['Origin'].str.lower() == ori_city)
    ]

    # If no matching rows are found
    if matching_fares.empty:
        return "No matching route found."

    # Extracting the relevant columns
    dist = matching_fares['Distance'].values[0]
    road = matching_fares['Road'].values[0]
    rail = matching_fares['Rail'].values[0]
    air = matching_fares['Air'].values[0]

    return {"Distance": dist, "Road": road, "Rail": rail, "Air": air}



col1, col2 = st.columns(2)
city = col1.text_input("Enter the Destination City")
ori_city = col2.text_input("Enter the Origin City")

st.empty()

col3, col4 = st.columns(2)
price = col3.number_input("Enter your preferable Price")
guest = col4.number_input("Enter the number of guest or rooms")


# # Example usage
# city_preference = 'Goa'
# price_preference = 2000
# guest_preference = 10

if st.button("Get Recommendation"):
    result = predict_hotel(city, price,guest)
    if isinstance(result, dict):
        with st.expander(f"**Recommended Hotel**"):
            st.write(f"**Hotel**: {result['hotel']}")
            st.write(f"**Address**: {result['address']}")
            st.write(f"**The** **Room** **Offers**: {result['room_features']}")
            st.write(f"**The Hotel offers Features as**: {result['hotel_features']}")

            # Fetch and display images of the recommended hotel
            images = get_hotel_image(result['hotel'])
            num_images = len(images)
            if num_images > 0:
                cols = st.columns(5)
                for idx, image in enumerate(images):
                    col = cols[idx % 5]
                    col.image(image['link'],use_column_width=True)

        places = get_places_to_visit(city)
        with st.expander(f"**Places to Visit**"):
            if isinstance(places, str):
                st.write(places)
            else:
                for place in places:
                    st.write(f"**{place}**")

        food_spots = get_food_spot(city)
        with st.expander(f"**Food** **Spot** **to** **try**"):
            if isinstance(food_spots, str):
                st.write(food_spots)
            else:
                for food_spot in food_spots:
                    st.write(f"**{food_spot}**")

        fares = get_fares(city, ori_city)
        if isinstance(fares, dict):
            with st.expander(f"**Travelling Fares**"):
                st.write(f"**Distance** **Between** **Cities**: {fares['Distance']} Km")
                st.write(f"**Road** **Fares**: {fares['Road']} Rs")
                st.write(f"**Rail** **Fares**: {fares['Rail']} Rs")
                st.write(f"**Air** **Fares**: {fares['Air']} Rs")

    else:
        st.write(result)

# predicted_hotel = predict_hotel(city, price,guest)
# address = get_hotel_info(predict_hotel)
# print(f'Predicted Hotel: {predicted_hotel}')