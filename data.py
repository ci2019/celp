"""
This file loads the data from the data directory and shows you how.
Feel free to change the contents of this file!
Do ensure these functions remain functional:
    - get_business(city, business_id)
    - get_reviews(city, business_id=None, user_id=None, n=10)
    - get_user(username)
"""

import os
import json
import random

DATA_DIR = "data"


def load_cities():
    """
    Finds all cities (all directory names) in ./data
    Returns a list of city names
    """
    return os.listdir(DATA_DIR)


def load(cities, data_filename):
    """
    Given a list of city names,
        for each city extract all data from ./data/<city>/<data_filename>.json
    Returns a dictionary of the form:
        {
            <city1>: [<entry1>, <entry2>, ...],
            <city2>: [<entry1>, <entry2>, ...],
            ...
        }
    """
    data = {}
    for city in cities:
        city_data = []
        with open(f"{DATA_DIR}/{city}/{data_filename}.json", "r") as f:
            for line in f:
                city_data.append(json.loads(line))
        data[city] = city_data
    return data


def get_business(city, business_id):
    """
    Given a city name and a business id, return that business's data.
    Returns a dictionary of the form:
        {
            name:str,
            business_id:str,
            stars:str,
            ...
        }
    """
    for business in BUSINESSES[city]:
        if business["business_id"] == business_id:
            return business
    raise IndexError(f"invalid business_id {business_id}")


def get_reviews(city, business_id=None, user_id=None, n=10):
    """
    Given a city name and optionally a business id and/or auser id,
    return n reviews for that business/user combo in that city.
    Returns a dictionary of the form:
        {
            text:str,
            stars:str,
            ...
        }
    """
    def should_keep(review):
        if business_id and review["business_id"] != business_id:
            return False
        if user_id and review["user_id"] != user_id:
            return False
        return True

    reviews = REVIEWS[city]
    reviews = [review for review in reviews if should_keep(review)]
    return random.sample(reviews, min(n, len(reviews)))


def get_user(username):
    """
    Get a user by its username
    Returns a dictionary of the form:
        {
            user_id:str,
            name:str,
            ...
        }
    """
    for _, users in USERS.items():
        for user in users:
            if user["name"] == username:
                return user
    raise IndexError(f"invalid username {username}")


CITIES = load_cities()
USERS = load(CITIES, "user")
BUSINESSES = load(CITIES, "business")
REVIEWS = load(CITIES, "review")
TIPS = load(CITIES, "tip")
CHECKINS = load(CITIES, "checkin")
