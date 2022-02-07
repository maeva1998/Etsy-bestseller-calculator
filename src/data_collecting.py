from tools import *

def main():
    """
    Function to retrive art-related listings from Etsy with HTTPS GET requests
    and update the csv file with more data
    :return: None
    """
    while True:
        try:
            get_data()
        except ValueError as e:
            print(e)
            break


if __name__ == "__main__":
    main()
