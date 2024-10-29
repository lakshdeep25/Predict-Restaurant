import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Dataset.csv')

# Visualizing geographical distribution of restaurants
plt.figure(figsize=(10, 6))
plt.scatter(data['Longitude'], data['Latitude'], alpha=0.5, c='blue', s=10)
plt.title('Geographical Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

# Grouping restaurants by city and analyzing concentration
city_counts = data['City'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=city_counts.index, y=city_counts.values, palette='viridis')
plt.title('Number of Restaurants by City')
plt.xticks(rotation=90)
plt.ylabel('Number of Restaurants')
plt.xlabel('City')
plt.show()

# Calculating statistics (Average Ratings and Price Range) by City
average_stats = data.groupby('City').agg({'Aggregate rating': 'mean', 'Price range': 'mean'}).reset_index()

# Visualizing average ratings by city
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='Aggregate rating', data=average_stats, palette='coolwarm')
plt.title('Average Ratings by City')
plt.xticks(rotation=90)
plt.ylabel('Average Rating')
plt.xlabel('City')
plt.show()

# Visualizing average price range by city
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='Price range', data=average_stats, palette='mako')
plt.title('Average Price Range by City')
plt.xticks(rotation=90)
plt.ylabel('Average Price Range')
plt.xlabel('City')
plt.show()
