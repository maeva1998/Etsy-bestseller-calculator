"""
File with all the tools and reusable function used all along the
project
"""

import requests as r
import pandas as pd
from enum import Enum
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords') uncomment if the package is not installed

# Etsy auth
app_name = "Maeva"
key_string = "sjmhnt35s8xtxyh6u3hvsgbv"
secret = "3n64el2462"


# column names for the raw tale (before cleaning and processing)
class RawLabels(Enum):
    TITLE = "title"
    CREATION_DATE = "original_creation_tsz"
    PRICE = "price"
    CURRENCY_CODE = "currency_code"
    TAXONOMY_ID = "taxonomy_id"
    TAXONOMY_PATH = "taxonomy_path"  # not present in the new data retrived
    MATERIALS = "materials"
    STYLE = "style"
    TAGS = "tags"
    FEATURED_RANK = "featured_rank"
    VIEWS = "views"
    NUM_FAVORERES = "num_favorers"
    WHO_MADE = "who_made"
    IS_CUSTOMIZABLE = "is_customizable"
    IS_DIGITAL = "is_digital"


# column names for the final tale (after cleaning and processing)
class Labels(Enum):
    PRICE = "price"
    TAXONOMY_ID = "taxonomy_id"
    FEATURED_RANK = "featured_rank"
    MATERIALS_DIST_FROM_TITLE = "materials_dist"
    STYLE_DIST_FROM_TITLE = "style_dist"
    TAGS_DIST_FROM_TITLE = "tags_dist"
    HANDMADE = "handmade"
    IS_CUSTOMIZABLE = "is_customizable"
    IS_DIGITAL = "is_digital"
    FEATURE = "features"
    SALEABILITY1 = "saleability_1"
    SALEABILITY2 = "saleability_2"


def get_art_taxonomy(dic, node):
    """
    The taxonomy is the classification system on Etsy.
    Evry field has his own unique taxonomy ID.
    The aim is to retrieve all the art-related taxonomy ID and update the taxonomy dictionary
    :param dic: Dictionary with the taxonomies id retrieved and their natural language
    classification. EX 123 : "glass art"
    :param node: art related root id (all the other IDs are his "children" and will be retrieved
    recursively thanks to this function)
    :return: None
    """
    dic[node["id"]] = node["name"]
    if len(node["children"]) == 0:
        pass
    else:
        for child in node["children"]:
            get_art_taxonomy(dic, child)


def get_data(table_path="./listing.csv"):
    """
    Get all the art-related listings and put it on a file
    :param table_path: path of the file were the data has to stored (by default is the listing.csv file)
    :return: None
    """
    # get all the taxonomy IDs and names
    url_tax = "https://openapi.etsy.com/v2/taxonomy/seller/get?api_key=" + key_string
    ans = r.request("get", url_tax)

    taxonomy_dic = {}
    node = ans.json()["results"][1]  # the second node is the art one
    get_art_taxonomy(taxonomy_dic, node)

    # get the listings data
    for tax_id in taxonomy_dic:
        listings_url = "https://openapi.etsy.com/v2/listings/active?taxonomy_id=" + str(
            tax_id) + "&api_key=" + key_string
        listings = r.request("get", listings_url)

        # retriving the english data
        for listing in listings.json()["results"]:
            try:
                if listing["language"] == "en-US" or listing["language"] == "en-GB":
                    file = open(table_path, "a")  # append data
                    file.write(
                        "\"" + str(listing["title"]) + "\"" + ";" +
                        str(listing["original_creation_tsz"]) + ";" +
                        str(listing["price"]) + ";" +
                        "\"" + str(listing["currency_code"]) + "\"" + ";" +
                        "\"" + str(listing["taxonomy_id"]) + "\"" + ";" +
                        "\"" + str(listing["taxonomy_path"]) + "\"" + ";" +
                        "\"" + str(listing["materials"]) + "\"" + ";" +
                        "\"" + str(listing["style"]) + "\"" + ";" +
                        "\"" + str(listing["tags"]) + "\"" + ";" +
                        "\"" + str(listing["featured_rank"]) + "\"" + ";" +
                        str(listing["views"]) + ";" +
                        str(listing["num_favorers"]) + ";" +
                        "\"" + str(listing["who_made"]) + "\"" + ";" +
                        "\"" + str(listing["is_customizable"]) + "\"" + ";" +
                        "\"" + str(listing["is_digital"]) + "\"" + "\n")
                    file.close()
            except UnicodeEncodeError as e:
                logging.debug(listing)
                logging.debug(str(e))
            except KeyError as e:
                logging.debug(listing)
                logging.debug(str(e))
            except ValueError as e:
                logging.debug(e)
                # timeout or connection errors
                time.sleep(60)


def get_raw_df(path="./listing.csv", labels=None):
    """
    Create a pandas dataframe from the csv raw table, and putting the rights labels
    :param path: path of the csv file with the data
    :return: the pandas dataframe obtained
    """
    if labels is None:
        labels = [k.value for k in RawLabels]
    df = pd.read_csv(filepath_or_buffer=path, encoding_errors="ignore", sep=";")
    df.columns = labels
    return df


def cleaning(string):
    """
    Tokenize, put in lower case, check that every world is alphanumeric and check that is not a stop-word
    :param string: string to clean
    :return: the clean string
    """
    token_list = word_tokenize(string)
    stemmer = SnowballStemmer("english")
    stop_w = set(stopwords.words('english'))
    lower_and_stem = [stemmer.stem(w.lower()) for w in token_list]
    clean_list = [w for w in lower_and_stem if (w not in stop_w and w.isalpha())]
    return " ".join(clean_list)


def cosine_dist(str1, str2):
    """
    Perform the cosine similarity between two strings
    :param str1: 1st string
    :param str2: 2nd string
    :return: float (the cosine)
    """
    if (str1 == []):
        return -2.0
    doc = [str("".join(str1)), str(str2)]
    try:
        vect = TfidfVectorizer().fit_transform(doc)
        cosine = cosine_similarity(vect)
        return float(cosine[0][1])
    except ValueError as e:
        return -1.0


def get_countvectorizer(tup):
    tup = eval(str(tup))  # be sure is a tuple
    res = []
    for i in range(int(tup[0])):
        if i in tup[1]:
            res.append(tup[2][tup[1].index(i)])
        else:
            res.append(0.0)
    return res
