from tools import *
import multiprocessing as mp
import numpy as np
from tqdm import *


def processing(raw_df):
    """
    Preprocessing and processing the raw data in order to
    obtain the final dataframe ready to be used for the
    ML part
    :param raw_df: raw DataFrame, by default is the data obtained via the function get_raw_data
    :return: the final DataFrame
    """
    print("Strarting processing...")
    # deleting rows that contains NaN values
    raw_df = raw_df.dropna(axis=0)

    # transform the style and material none column with an empty list
    tqdm.pandas(desc="treanform NaN values into empty list -> []")
    raw_df[RawLabels.MATERIALS.value] = raw_df[RawLabels.MATERIALS.value].progress_apply(
        lambda x: "[]" if (x is None or x == "") else x)
    raw_df[RawLabels.STYLE.value] = raw_df[RawLabels.STYLE.value].progress_apply(
        lambda x: "[]" if (x is None or x == "") else x)
    raw_df[RawLabels.TAGS.value] = raw_df[RawLabels.TAGS.value].progress_apply(
        lambda x: "[]" if (x is None or x == "") else x)

    # transform the who_made column in true or false (1 or 0) for "handmade" "not handmade"
    tqdm.pandas(desc="transform who_made column in a true/false column")
    raw_df[Labels.WHO_MADE.value] = raw_df[Labels.WHO_MADE.value].progress_apply((lambda x: 1 if x == "i_did" else 0))

    # add a normalized views/creation_time column and num_favorers/creation_time columns (sellabilities values)
    tqdm.pandas(desc="Creation of sellability columns")
    raw_df["views_time_ratio"] = \
        raw_df[RawLabels.VIEWS.value] / raw_df[Labels.CREATION_DATE.value]
    raw_df["favorers_time_ratio"] = \
        raw_df[RawLabels.NUM_FAVORERES.value] / raw_df[RawLabels.NUM_FAVORERES.value]
    max_ratio_1 = raw_df["views_time_ratio"].max()
    max_ratio_2 = raw_df["favorers_time_ratio"].max()
    raw_df[Labels.SELLABILITY1.value] = raw_df.progress_apply(
        (lambda x: (x["views_time_ratio"] * 100) / max_ratio_1), axis=1)
    raw_df[Labels.SELLABILITY2.value] = raw_df.progress_apply(
        (lambda x: (x["favorers_time_ratio"] * 100) / max_ratio_2), axis=1)

    # create a price column with the same currency price for every item
    tqdm.pandas(desc="Set the same price for all listing")
    raw_df[Labels.PRICE.value] = raw_df.progress_apply((lambda x: x[Labels.PRICE.value] if (
            [RawLabels.CURRENCY_CODE.value] != "GBP") else x[Labels.PRICE.value] * 0.75), axis=1)
    raw_df.drop(RawLabels.CURRENCY_CODE.value, axis=1)

    # add a column for the views * price (to represent the possible earned money)
    # in average only 20 to 40% of people buy what the put in their cart (so we go for a 30%)
    tqdm.pandas(desc="Creation of possible income column")
    raw_df[Labels.POSSIBLE_SELL_INCOME.value] = raw_df.progress_apply(
        (lambda x: x[RawLabels.NUM_FAVORERES.value] * 0.3 * x[RawLabels.PRICE.value]), axis=1)

    # clean the tags and title from stop_words, bad encoded words, non alpha terms and tokenize
    tqdm.pandas(desc="Preprocessing title")
    raw_df[RawLabels.TITLE.value] = raw_df[RawLabels.TITLE.value].progress_apply(lambda x: cleaning(x))
    tqdm.pandas(desc="Preprocessing tags")
    raw_df[RawLabels.TAGS.value] = raw_df[RawLabels.TAGS.value].progress_apply(
        lambda tags_list: [cleaning(x) for x in tags_list.split(",")])
    tqdm.pandas(desc="Preprocessing material list")
    raw_df[RawLabels.MATERIALS.value] = raw_df[RawLabels.MATERIALS.value].progress_apply(
        lambda mat_list: [cleaning(x) for x in mat_list.split(",")])
    tqdm.pandas(desc="Preprocessing style")
    raw_df[RawLabels.STYLE.value] = raw_df[RawLabels.STYLE.value].progress_apply(
        lambda style_list: [cleaning(x) for x in style_list.split(",")])

    # create a column with -2 if the tags are not presents and the cosine
    # distance between tags and title if tags are present (same with material and style)
    # title, tags, material and style share almost the same words. So we have some information
    # redundancy/multicollinearity if we choose to keep all of them.
    # This changement allow us to know if the tag is present or not and is
    tqdm.pandas(desc="Computing cosine distance for tags")
    raw_df[Labels.TAGS_DIST_FROM_TITLE.value] = \
        raw_df.progress_apply(lambda x: -1 if (x[RawLabels.TAGS.value]
                                               == "[]" or x[RawLabels.TAGS.value] == "") else cosine_dist(
            " ".join(x[RawLabels.TAGS.value]),
            x[RawLabels.TITLE.value]), axis=1)

    tqdm.pandas(desc="Computing cosine distance for materials")
    raw_df[Labels.MATERIALS_DIST_FROM_TITLE.value] = \
        raw_df.progress_apply(lambda x: -1 if x[RawLabels.MATERIALS.value] == [] else cosine_dist(
            " ".join(x[RawLabels.MATERIALS.value]), x[RawLabels.TITLE.value]), axis=1)

    tqdm.pandas(desc="Computing cosine distance for styles")
    raw_df[Labels.STYLE_DIST_FROM_TITLE.value] = \
        raw_df.progress_apply(
            lambda x: -1 if x[RawLabels.STYLE.value] == [] else cosine_dist(" ".join(x[RawLabels.STYLE.value]),
                                                                            x[RawLabels.TITLE.value]), axis=1)

    df1 = add_tfid(raw_df[RawLabels.TITLE.value])
    final_df = pd.concat([df1, raw_df], axis=1)\
        .drop(RawLabels.TAXONOMY_PATH.value, axis=1)\
        .drop(RawLabels.VIEWS.value, axis=1)\
        .drop(RawLabels.NUM_FAVORERES.value, axis=1)\
        .drop("views_time_ratio", axis=1)\
        .drop("favorers_time_ratio", axis=1)\
        .drop(RawLabels.MATERIALS.value, axis=1)\
        .drop(RawLabels.TAGS.value, axis=1)\
        .drop(RawLabels.STYLE.value, axis=1)\
        .drop(RawLabels.TITLE.value, axis=1)

    print("Done with the preprocesing")

    return final_df


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = mp.Pool(mp.cpu_count())
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == "__main__":
    raw_df = get_raw_df()
    df = parallelize_dataframe(raw_df, processing)
    print(len(df))
    print(len(df.columns))
    print(df.columns)
    print(df.head(10))
