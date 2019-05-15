from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS, get_business, get_user, get_reviews
from pandas import Series, DataFrame

import random
import pandas as pd
import numpy as np


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
    for city, users in USERS.items():
        for user in users:
            all_user_ids.append(user['user_id'])

    df_users = pd.DataFrame(index=all_user_ids, columns=['user_id', 'name'])

    for city, users in USERS.items():
        for user in users:
            df_users['user_id'][user['user_id']] = user['user_id']
            df_users['name'][user['user_id']] = user['name']

    df_users = df_users.set_index('user_id')
    return df_users


def make_rating_matrix():
    # Returns dataframe with rating stars and info
    ratings_list = []
    for i, j in REVIEWS.items():
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
    # get movie and user id's
    business_ids = ratings['business_id'].unique()
    user_ids = ratings['user_id'].unique()

    # create empty data frame
    pivot_data = pd.DataFrame(np.nan, columns=user_ids,
                              index=business_ids, dtype=float)

    # use the function get_rating to fill the matrix
    a = 0
    for business in business_ids:
        for user in user_ids:
            a = a+1
            print(a)
            pivot_data[user][business] = get_rating(ratings, user, business)

    return pivot_data


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

    utility_matrix = pivot_ratings(make_rating_matrix().head())
    print(utility_matrix)

    if not city:
        city = random.choice(CITIES)
    return random.sample(BUSINESSES[city], n)
