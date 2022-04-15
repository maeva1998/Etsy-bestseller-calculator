import unittest
from pyspark.sql import SparkSession

from pyspark.ml.feature import HashingTF, IDF, Tokenizer

class MyTestCase(unittest.TestCase):
    def test_something(self):
        spark = SparkSession.builder.getOrCreate()
        sentenceData = spark.createDataFrame([
            (0.0, "Hi I heard about Spark"),
            (0.0, "I wish Java could use case classes"),
            (1.0, "Logistic regression models are neat")
        ], ["label", "sentence"])

        tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
        wordsData = tokenizer.transform(sentenceData)

        hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
        featurizedData = hashingTF.transform(wordsData)
        # alternatively, CountVectorizer can also be used to get term frequency vectors

        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idfModel = idf.fit(featurizedData)
        rescaledData = idfModel.transform(featurizedData)

        rescaledData.select("label", "features").show()


if __name__ == '__main__':
    unittest.main()
