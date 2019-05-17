from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS, get_business, get_user
from pandas import Series, DataFrame
from math import sqrt

import sklearn.metrics.pairwise as pw
import random
import pandas as pd
import numpy as np


def make_business_matrix(city):
    # Returns dataframe with business info
    all_ids = [business["business_id"] for business in BUSINESSES[city]]
    df_business = pd.DataFrame(index=all_ids, columns=['business_id', 'name', 'address', 'city', 'state', 'postal_code',
                                                       'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours'])
    for business in BUSINESSES[city]:
        b_id = business["business_id"]
        for a in business:
            df_business[a][b_id] = business[a]

    # df_business = df_business.set_index('business_id')
    return df_business


def make_user_matrix():
    # Returns dataframe with user name and id
    all_user_ids = []
    for _, users in USERS.items():
        for user in users:
            all_user_ids.append(user['user_id'])

    df_users = pd.DataFrame(index=all_user_ids, columns=['user_id', 'name'])

    for _, users in USERS.items():
        for user in users:
            df_users['user_id'][user['user_id']] = user['user_id']
            df_users['name'][user['user_id']] = user['name']

    df_users = df_users.set_index('user_id')
    return df_users


def make_rating_matrix():
    # Returns dataframe with rating stars and info
    ratings_list = []
    for _, j in REVIEWS.items():
        for k in j:
            ratings_list.append(
                {'user_id': k['user_id'], 'business_id': k['business_id'], 'rating': k['stars']})

    df_ratings = pd.DataFrame(ratings_list, columns=[
                              'user_id', 'business_id', 'rating'])
    return df_ratings


def number_of_businesses(ratings):
    """Determine the number of unique movie id's in the data.

    Arguments:
    ratings -- a dataFrame containing a column 'movieId'
    """
    return len(ratings['business_id'].unique())


def number_of_users(ratings):
    """Determine the number of unique user id's in the data.

    Arguments:
    ratings -- a dataFrame containing a column 'userId'
    """
    return len(ratings['user_id'].unique())


def number_of_ratings(ratings):
    """Count the number of ratings of a dataset.

    Arguments:
    ratings -- a dataFrame.
    """
    return ratings.shape[0]


def rating_density(ratings):
    """Compute the ratings given a dataset.

    Arguments:
    ratings -- a dataFrame contasining the columns 'userId' and 'movieId'
    """
    return number_of_ratings(ratings) / (number_of_businesses(ratings) * number_of_users(ratings))


def split_data(data, d=0.75):
    """Split data in a training and test set.

    Arguments:
    data -- any dataFrame.
    d    -- the fraction of data in the training set
    """
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]


def pivot_ratings(df):
    """Creates a utility matrix for user ratings for movies

    Arguments:
    df -- a dataFrame containing at least the columns 'movieId' and 'genres'

    Output:
    a matrix containing a rating in each cell. np.nan means that the user did not rate the movie
    """
    df = df.reset_index(drop=True)
    return df.pivot_table(values='rating', columns='user_id', index='business_id')


def create_similarity_matrix_cosine(matrix):
    """Creates a adjusted(/soft) cosine similarity matrix.

    Arguments:
    matrix -- a utility matrix

    Notes:
    Missing values are set to 0. This is technically not a 100% correct, but is more convenient
    for computation and does not have a big effect on the outcome.
    """
    mc_matrix = matrix - matrix.mean(axis=0)
    return pd.DataFrame(pw.cosine_similarity(mc_matrix.fillna(0)), index=matrix.index, columns=matrix.index)


def predict_ratings(similarity, utility, to_predict):
    """Predicts the predicted rating for the input test data.

    Arguments:
    similarity -- a dataFrame that describes the similarity between items
    utility    -- a dataFrame that contains a rating for each user (columns) and each movie (rows).
                  If a user did not rate an item the value np.nan is assumed.
    to_predict -- A dataFrame containing at least the columns movieId and userId for which to do the predictions
    """
    # copy input (don't overwrite)
    ratings_test_c = to_predict.copy()
    # apply prediction to each row
    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(
        similarity, utility, row['user_id'], row['business_id']), axis=1)
    return ratings_test_c

### Helper functions for predict_ratings_item_based ###


def predict_ids(similarity, utility, userId, itemId):
    # select right series from matrices and compute
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:, userId], similarity[itemId])
    return 0


def predict_vectors(user_ratings, similarities):
    # select only movies actually rated by user
    relevant_ratings = user_ratings.dropna()

    # select corresponding similairties
    similarities_s = similarities[relevant_ratings.index]

    # select neighborhood
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]

    # if there's nothing left return a prediction of 0
    norm = similarities_s.sum()
    if(norm == 0):
        return 0

    # compute a weighted average (i.e. neighborhood is all)
    return np.dot(relevant_ratings, similarities_s)/norm


def mse(predicted_ratings):
    """Computes the mean square error between actual ratings and predicted ratings

    Arguments:
    predicted_ratings -- a dataFrame containing the columns rating and predicted rating
    """
    diff = predicted_ratings['rating'] - predicted_ratings['predicted rating']
    return (diff**2).mean()


def extract_genres(movies):
    """Create an unfolded genre dataframe. Unpacks genres seprated by a '|' into seperate rows.

    Arguments:
    movies -- a dataFrame containing at least the columns 'movieId' and 'genres' 
              where genres are seprated by '|'
    """
    genres_m = movies.apply(lambda row: pd.Series(
        [row['business_id']] + row['categories'].lower().split(",")), axis=1)
    stack_genres = genres_m.set_index(0).stack()
    df_stack_genres = stack_genres.to_frame()
    df_stack_genres['business_id'] = stack_genres.index.droplevel(1)
    df_stack_genres.columns = ['categorie', 'business_id']
    return df_stack_genres.reset_index()[['business_id', 'categorie']]


def pivot_categories(df):
    """Create a one-hot encoded matrix for genres.

    Arguments:
    df -- a dataFrame containing at least the columns 'movieId' and 'genre'

    Output:
    a matrix containing '0' or '1' in each cell.
    1: the movie has the genre
    0: the movie does not have the genre
    """
    return df.pivot_table(index='business_id', columns='categorie', aggfunc='size', fill_value=0)


def create_similarity_matrix_categories(matrix):
    """Create a  """
    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index=matrix.index, columns=matrix.index)


def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """

    # Recommendation for business page
    if business_id:
        print("Business id found!", business_id)
        a = pivot_categories(extract_genres(make_business_matrix(city)))
        df_similarity_genres = create_similarity_matrix_categories(a)
        df_categories = df_similarity_genres.sort_values(
            by=[business_id], ascending=False)[business_id].head(11)
        print(df_categories)
        # Get top ten businesses with info
        top_ten = [get_business(city, i) for i in df_categories.index.values]
        top_ten.pop(0)

        # MSE testing
        _, df_ratings_test = split_data(
            make_rating_matrix(), d=0.9)
        utility_matrix = pivot_ratings(make_rating_matrix())
        predicted_ratings = predict_ratings(df_similarity_genres, utility_matrix, df_ratings_test[[
                                            'user_id', 'business_id', 'rating']])
        mse_genres = mse(predicted_ratings)
        print("MSE content based:", mse_genres)

        return top_ten

    # Recommendation for home page of user
    if not city:
        city = random.choice(CITIES)

    # If user is not logged in return random business
    if not user_id:
        return random.sample(BUSINESSES[city], n)

    utility_matrix = pivot_ratings(make_rating_matrix())
    similarity = create_similarity_matrix_cosine(utility_matrix)
    utility_matrix_copy = utility_matrix.copy()

    # MSE testing
    _, df_ratings_test = split_data(
        make_rating_matrix(), d=0.9)
    df_predicted_cf_item_based = predict_ratings(
        similarity, utility_matrix, df_ratings_test[['user_id', 'business_id', 'rating']])
    mse_item = mse(df_predicted_cf_item_based)
    print("MSE item based:", mse_item)

    # Predict all ratings
    all_business_ids = [business["business_id"]
                        for business in BUSINESSES[city]]
    all_user_ids = []
    for _, users in USERS.items():
        for user in users:
            all_user_ids.append(user['user_id'])

    for user in all_user_ids:
        for business in all_business_ids:
            utility_matrix_copy[user][business] = predict_ids(
                similarity, utility_matrix, user, business)

    # Remove already rated business
    already_rated = utility_matrix[user_id].dropna()
    df = utility_matrix_copy.sort_values(
        by=[user_id], ascending=False)[user_id].drop(already_rated.index.values).head(10)

    # Get top ten businesses with info
    top_ten = [get_business(city, i) for i in df.index.values]

    return top_ten
