import requests
import sys
from bs4 import BeautifulSoup

class Book():

    def __init__(self) -> None:

        print("Initilizing book")

        self.url = None
        self.raw_text = None
        self.tokenize_method = None
        self.normalize_method = None

    def print_info_by_attr(self, attribute_name: str):

        print(getattr(self, attribute_name, "Attribute not found"))
        return
    

    def __parse_metadata(self):
        """
            Parses metadata from gutenberg info
        """
        pass

    def get_book(self, url: str ):
        """
            get_book: Using given URL, attempt to get text

                Only works with Project Guttenburg since this is a class project

            Parameters:
                url: Url to guttenberg page containing ebook text
            
            return: int: 0 for success, 1 for failure
        """
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            self.raw_text = soup.text
            return 0
        else:
            print("Error: Failed to retrieve the webpage", file=sys.stderr)
            return 1

    def set_tokenization_method(self):
        pass

    def set_normalization_method(self):
        pass

    def tokenize(self):
        pass

    def normalize(self):
        pass
