from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS, get_business, get_user, get_reviews
from pandas import Series, DataFrame
from math import sqrt

import random
import pandas as pd
import numpy as np


def square(a):
    return a*a


def make_business_matrix(city):
    # Returns dataframe with business info
    all_ids = [business["business_id"] for business in BUSINESSES[city]]
    df_business = pd.DataFrame(index=all_ids, columns=['business_id', 'name', 'address', 'city', 'state', 'postal_code',
                                                       'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours'])
    for business in BUSINESSES['city']:
        b_id = business["business_id"]
        for a in business:
            df_business[a][b_id] = business[a]

    df_business = df_business.set_index('business_id')
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


def get_rating(ratings, user_id, business_id):
    """Given a userId and movieId, this function returns the corresponding rating.
       Should return NaN if no rating exists."""
    rating = ratings[ratings["user_id"] ==
                     user_id][ratings["business_id"] == business_id].rating
    if len(rating) == 0:
        return np.nan
    else:
        return rating.values[0]


def pivot_ratings(ratings):
    ratings = ratings.reset_index(drop=True)
    """ takes a rating table as input and computes the utility matrix """
    # get business and user id's
    business_ids = ratings['business_id'].unique()
    user_ids = ratings['user_id'].unique()

    # create empty data frame
    pivot_data = pd.DataFrame(np.nan, columns=user_ids,
                              index=business_ids, dtype=float)

    # use the function get_rating to fill the matrix
    for business in business_ids:
        for user in user_ids:
            pivot_data[user][business] = get_rating(ratings, user, business)

    return pivot_data


def get_points(matrix, id1):
    matrix = matrix.fillna(0)
    return matrix[matrix.index == id1].values[0]


def cosine_similarity(matrix, id1, id2):
    """Compute cosine similarity between two rows."""

    # get all ratings for the point
    selected_features = matrix.loc[id1].notna() & matrix.loc[id2].notna()

    # if no matching features, return NaN
    if not selected_features.any():
        return 0.0

    # get the features from the matrix
    features1 = matrix.loc[id1][selected_features]
    features2 = matrix.loc[id2][selected_features]

    if len(features1) == 0 or len(features2) == 0:
        return 0

    # check if points are the same
    if np.array_equal(features1, features2):
        return 1

    # calculate range for for loop
    total_range = 0
    if len(features1) == 0:
        total_range = len(features2)
    else:
        total_range = len(features1)

    # calculate numerator
    numerator = 0
    numerator = (features1 * features2).sum()

    # calculate denominator
    denominator = 0
    features1_copy = [square(x) for x in features1]
    features2_copy = [square(x) for x in features2]
    denominator_left = sqrt(sum(features1_copy))
    denominator_right = sqrt(sum(features2_copy))
    denominator = denominator_left * denominator_right

    # check if denominator is valid
    if denominator == 0 or denominator is np.nan:
        return 0.0

    # calculate similarity
    similarity = numerator / denominator

    # check if similarity is number
    if similarity is np.nan:
        return 0.0

    # else return similarity
    return similarity


def create_similarity_matrix_cosine(matrix):
    """ creates the similarity matrix based on cosine similarity """
    similarity_matrix = pd.DataFrame(
        0, index=matrix.index, columns=matrix.index, dtype=float)
    for i in matrix.index:
        for j in matrix.index:
            similarity_matrix[i][j] = cosine_similarity(matrix, i, j)
    return similarity_matrix


def calculate_mean(matrix, id1):
    matrix = matrix
    return np.mean(matrix[id1])


def mean_center_columns(matrix):
    centered_matrix = matrix.copy()
    for i in matrix.columns.values:
        mean = calculate_mean(matrix, i)
        centered_matrix[i] = centered_matrix[i] - mean
    return centered_matrix


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

    utility_matrix = pivot_ratings(make_rating_matrix())
    centered_utility_matrix = mean_center_columns(utility_matrix)
    print(centered_utility_matrix)

    similarity = create_similarity_matrix_cosine(centered_utility_matrix)
    print(similarity)

    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)
