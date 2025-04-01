import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import requests
import zipfile
import io
import os

#this is a function that loads the MovieeLens 100k dataset
#from the official datasets's external website
# it will unpack the downloaded .zip file
def download_and_extract_movielens():
    if not os.path.exists('ml-100k'):
        print("Downloading MovieLens 100k dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print("MovieLens 100k dataset downloaded and extracted successfully.")
    
    else:
        print("The Datasets already exists.Download skipped.")

#we gonna define a new function and load by calling the new function,
#and put the data into a Pandas Dataframe and get some basic information about it

download_and_extract_movielens()

ratings_df = pd.read_csv('ml-100k/u.data' ,sep='/t',
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
print(f"Dataset shape: {ratings_df.shape}")
print(f"Number of unique users: {ratings_df['user_id'].nunique()}")
print(f"Range of ratings: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")

#now we gonna pack the datasets into an easily manageable formart by the library's implementation of matrix factorization
#technique.
#we gonna split the data into training and test sets for model evaluation
#notice the importance of specifying the correct range of numerical ratings in the dataset when initializing
#the Reader object

reader = Reader(rating_scale=(1-5))
data = Dataset.load_from_df(Ratings_df[['user_id', 'item_id', 'rating']], reader)

trainset, testset= train_test_split(data, test_size=0.2, random_state=42)

#now we gonna initialize, train and evaluate the matric factorization model, we will use Singular value decomposition(SVD)

model =SVD(n_factors=20, lr_all=0.01, reg_all=0.01, n_epochs=20, random_state=42)
model.fit(trainset)

predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

#for more robust evaluation, we optionally applay cross-validation
cv_results = cross_validate(model, data, meeasures=['RMSE','MAE'], cv=5, verbose=True)

print(f'Avarage RMSE: {cv_results['test-rmse'].mean():.4f}')
print(f'Avarage MAE: {cv_results['test-mae'].mean():.4f}')

#now we are using it seeing it inAction THE MOVIE RECOMMENDER SYSTEM
#defining 2 custom functions: one loads set of of movie titles, 2 given a user ID and a Number N of desired Recommendations
#we use the model to obtain the list of top-N recommended movies for that user, based on their preferences reflected
#in the original ratings data, The Latter function is the most insightful part of the entire code,inline comments have been added for better
#understanding of the process involved

def get_movie_names():
    movies_df =pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                           header=None, usecols=[0,1], names=['item_id','title'])
    
    return movies_df
movies_df = get_movies_names()

def recommend_movies(user_id, n=10):
    #List of all movies
    all_movies = movies_df['item_id'].unique()
    
    #movies already rrated by user
    rated_movies = ratings_df[ratings_df['user_id'] == user_id]['item_id'].values
    
    #movies not yet rated by the user
    unrated_movies = np.setdiff1d(all_movies, rated_movies)
    
    #predicting ratings on unseen movies, by usingthe trainend SVD model
    predictions=[]
    for item_id in unrated_movies:
        predicted_rating = model.predict(user_id, item_id).est
        predictions.append((item_id, predicted_rating))
        
    #rank predictions by estimated rating
        predictions.sort(key = lambda x: x[1], reverse = True)
        
    #Get top N recommendations
        top_recommendations = predictions[:n]
        
    #fetch movie titles associated  with top N recommendations
        recommendations = pd.DataFrame(top_recommendations, columns=['item_id', 'predicted_rating'])
        recommendations = recommendations.merge(movies_df, on = 'item_id')
    
    return recommendations

user_id= 42
recommendations = recommend_movies(user_id,n=10)
print(f"/Top 10 recommended movies for user  {user_id}:")
print(recommendations[['Title', 'predicted_rating']])


