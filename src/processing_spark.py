# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer

from pyspark.sql.functions import *
from pyspark.sql.types import *

from tools import *
import os

# TODO eventuellement au lieu de la distance au titre il faudra performer un countvectorizer binaire (pour titre
#  style etc)

def processing(path = "./listing.csv"):
    # attention it's important to set this!!!
    os.environ['PYSPARK_PYTHON'] = "python"  # sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = "python"  # sys.executable

    # Create spark session and loading the table
    spark = SparkSession.builder.getOrCreate()

    raw_labels = [k.value for k in RawLabels]
    raw_df = spark.read.csv(path=path, sep=";").toDF(*raw_labels)

    # starting the preocessing

    # selecting only USD, GBP and EUR as currencies
    new_df = raw_df.where((col(RawLabels.CURRENCY_CODE.value) == "USD")
                          | (col(RawLabels.CURRENCY_CODE.value) == "GBP")
                          | (col(RawLabels.CURRENCY_CODE.value) == "EUR"))\

    new_df = new_df.withColumn(Labels.FEATURED_RANK.value, when(col(Labels.FEATURED_RANK.value) == "None", -1)
                               .otherwise(col(Labels.FEATURED_RANK.value)))

    # transforming the categorical and bool column (string) into a binary (numerical) one
    new_df = new_df.withColumn(Labels.HANDMADE.value,
                               when(col(RawLabels.WHO_MADE.value) == "i_did", lit(1)).otherwise(lit(0))) \
        .withColumn(Labels.IS_DIGITAL.value, when(col(Labels.IS_DIGITAL.value) == "True", 1).otherwise(0)) \
        .withColumn(Labels.IS_CUSTOMIZABLE.value, when(col(Labels.IS_CUSTOMIZABLE.value) == "True", 1).otherwise(0))

    # Cleaning
    new_df = new_df.withColumn(RawLabels.TITLE.value, udf(cleaning, StringType())(col(RawLabels.TITLE.value))) \
        .withColumn(RawLabels.TAGS.value, udf(cleaning, StringType())(col(RawLabels.TAGS.value))) \
        .withColumn(RawLabels.MATERIALS.value, udf(cleaning, StringType())(col(RawLabels.MATERIALS.value))) \
        .withColumn(RawLabels.STYLE.value, udf(cleaning, StringType())(col(RawLabels.STYLE.value)))

    # computing the cosine distancy between the title and all the other info
    new_df = new_df.withColumn(Labels.TAGS_DIST_FROM_TITLE.value,
                               udf(cosine_dist, FloatType())(col(RawLabels.TAGS.value), col(RawLabels.TITLE.value))) \
        .withColumn(Labels.MATERIALS_DIST_FROM_TITLE.value,
                    udf(cosine_dist, FloatType())(col(RawLabels.MATERIALS.value), col(RawLabels.TITLE.value))) \
        .withColumn(Labels.STYLE_DIST_FROM_TITLE.value,
                    udf(cosine_dist, FloatType())(col(RawLabels.STYLE.value), col(RawLabels.TITLE.value)))

    new_df = new_df.withColumn(RawLabels.TITLE.value, split(col(RawLabels.TITLE.value), " ") )

    # Fit a CountVectorizerModel from the corpus
    cv = CountVectorizer(inputCol=RawLabels.TITLE.value, outputCol=Labels.FEATURE.value)
    model = cv.fit(new_df)
    new_df = model.transform(new_df)

    #getting the bag of words vector
    new_df = new_df.withColumn(Labels.FEATURE.value, udf(get_countvectorizer, ArrayType(FloatType()))(col(Labels.FEATURE.value)))

    # columns useful for the rest of the computation (to drop at the end)
    new_df = new_df.withColumn("views_time_ratio", col(RawLabels.VIEWS.value) / col(RawLabels.CREATION_DATE.value)) \
        .withColumn("favorers_time_ratio", col(RawLabels.NUM_FAVORERES.value) / col(RawLabels.CREATION_DATE.value))

    # computing saleability columns and computing the price in american dollars
    new_df = new_df.withColumn(Labels.SALEABILITY1.value,
                               round(
                                   col("views_time_ratio") * 100 / new_df.agg({"views_time_ratio": "max"}).collect()[0][
                                       0],
                                   1)) \
        .withColumn(Labels.SALEABILITY2.value,
                    round(col("favorers_time_ratio") * 100 / new_df.agg({"favorers_time_ratio": "max"}).collect()[0][0],
                          1)) \
        .withColumn(Labels.PRICE.value,
                    when(col(RawLabels.CURRENCY_CODE.value) == "GBP", round(col(Labels.PRICE.value) * 1.36, 2)) \
                    .when(col(RawLabels.CURRENCY_CODE.value) == "USD", round(col(Labels.PRICE.value), 2)) \
                    .otherwise(round(col(Labels.PRICE.value) * 1.13, 2)))

    # drop useless column
    new_df = new_df.drop(str(RawLabels.WHO_MADE.name)) \
        .drop("views_time_ratio") \
        .drop("favorers_time_ratio") \
        .drop(str(RawLabels.CREATION_DATE.value)) \
        .drop(str(RawLabels.VIEWS.value)) \
        .drop(str(RawLabels.NUM_FAVORERES.value)) \
        .drop(str(RawLabels.TITLE.value)) \
        .drop(str(RawLabels.CURRENCY_CODE.value)) \
        .drop(str(RawLabels.STYLE.value)) \
        .drop(str(RawLabels.TAGS.value)) \
        .drop(str(RawLabels.MATERIALS.value)) \
        .drop(str(RawLabels.TAXONOMY_PATH.value))\
        .drop(str(RawLabels.CREATION_DATE))

    X = new_df.select(Labels.PRICE.value,
                      Labels.TAXONOMY_ID.value,
                      Labels.FEATURED_RANK.value,
                      Labels.MATERIALS_DIST_FROM_TITLE.value,
                      Labels.STYLE_DIST_FROM_TITLE.value,
                      Labels.TAGS_DIST_FROM_TITLE.value,
                      Labels.HANDMADE.value,
                      Labels.IS_CUSTOMIZABLE.value,
                      Labels.IS_DIGITAL.value,
                      Labels.FEATURE.value)

    Y = new_df.select(Labels.SALEABILITY1.value)
    Y2 = new_df.select(Labels.SALEABILITY2.value)
    return X, Y, Y2


processing()