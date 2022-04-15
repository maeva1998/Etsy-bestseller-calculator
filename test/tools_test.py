import unittest
from src import tools as t
import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_cleaning_function(self):
        string = "Hi, how are you? how are things  going? I feel sick but happy"
        clean_string = "hi thing go feel sick happi"

        string2 = "Valentines Wine SVG, Valentine&#39;s Day Wine Bag SVG, Valentines Wine Glass SVG, Valentines Wine " \
                  "Label Png Svg, Wine Tumbler Svg, Digital Files "
        clean_string2 = "valentin wine svg valentin day wine bag svg valentin wine glass svg valentin wine label png " \
                        "svg wine tumbler svg digit file "

        self.assertEqual(clean_string, t.cleaning(string))
        self.assertEqual(clean_string2, t.cleaning(string2))

    def test_cleaning_function(self):
        string = "['exhibition poster', 'exhibition print', 'exhibition art', 'gallery print', 'Art print', " \
                 "'impressionism print', 'Flowers poster', 'floral pattern', 'Botanical ', 'Kitchen ', " \
                 "'Vintage flowers', 'Scandinavian ', 'Floral print '] "
        clean_string = "exhibit poster exhibit print exhibit art galleri print art print impression print flower " \
                       "poster floral pattern botan kitchen vintag flower scandinavian floral print"

        self.assertEqual(clean_string, t.cleaning(string))

    def test1_normal_cosine_funct(self):
        str1 = "I love love love cats"
        str2 = "I hate dogs"

        self.assertLessEqual(t.cosine_dist(str1, str2), 0.5)

    def test2_cosine_funct(self):
        str1 = "I love love love cats"
        str2 = "I love dogs"

        self.assertGreaterEqual(t.cosine_dist(str1, str2), 0.5)


    def test_error_cosine_funct(self):
        str3 = "I love cat"
        str4 = 1

        self.assertEqual(t.cosine_dist(str3, str4), 0)

    def test2_error_cosine_funct(self):
        str4 = ""
        self.assertEqual(t.cosine_dist(str4, str4), -1)

    def test_get_raw_df(self):
        labels = ["Name", "Age", "Hobbies"]
        df = t.get_raw_df("./listing_test.txt", labels)
        print(df)

    def test_get_raw_df_error(self):
        labels = ["Name", "Age", "Hobbies"]
        self.assertRaises(ValueError, t.get_raw_df, "./listing_test_error.txt", labels)


    def test_adding_tfid(self):
        df = pd.read_table("test_df.txt", sep=",")
        col = df["txt"]
        df1 = t.add_tfid(col)
        final_df = pd.concat([df, df1], axis=1).drop("txt", axis=1)
        print(final_df)
        print(len(final_df))
        print(len(final_df.columns))

    def get_countvectorizer_test(self):
        tup = (5, [3,2], [1,5])
        res = t.get_countvectorizer(tup)
        self.assertEqual(res, [0, 0, 5, 1, 0])


if __name__ == '__main__':
    unittest.main()
