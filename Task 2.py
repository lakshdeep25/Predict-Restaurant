import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('Dataset.csv')

# Dropping unnecessary columns for prediction
data_clean = data.drop(['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Rating color', 'Rating text'], axis=1)

# Handling missing values
data_clean.fillna(method='ffill', inplace=True)

# Defining the features for recommendations (Cuisines, Price range, Aggregate rating)
data_recommend = data_clean[['Cuisines', 'Price range', 'Aggregate rating']]

# Encoding the 'Cuisines' column using LabelEncoder
label_encoder = LabelEncoder()
data_recommend['Cuisines'] = label_encoder.fit_transform(data_recommend['Cuisines'])

# Compute the similarity between restaurants using cosine similarity
similarity_matrix = cosine_similarity(data_recommend)

# Function to recommend restaurants based on a given restaurant index
def recommend_restaurant(restaurant_index, num_recommendations=5):
    similarity_scores = list(enumerate(similarity_matrix[restaurant_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sorted_scores[1:num_recommendations+1]]
    
    return data.iloc[recommended_indices][['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating']]

# Example: Recommending restaurants similar to restaurant at index 0
recommendation = recommend_restaurant(restaurant_index=0, num_recommendations=5)
print(recommendation)
