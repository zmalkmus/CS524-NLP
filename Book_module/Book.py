import requests
import sys
import re
from bs4 import BeautifulSoup

from unidecode import unidecode

import spacy
import string 

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure to download the required resources
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('punkt_tab')

class Book():

    def __init__(self, book_number: int) -> None:
        """
            Book_number needs to be between [1-3]
        """
        print(f"[__init__]: Initilizing book: {book_number}")
        self.book_num = book_number
        self.url = None
        self.raw_text = None
        self.tokenize_method = None
        self.normalize_method = None
        self.raw_word_vec = None
        self.chapters = list() # Holds a string for each of the chapters
        self.paragraphs = list() # Holds lists of strings for each paragraph in each chapter
        self.sentences = list()
        self.pre_process_string = str()
        self.names = list()
        self.full_text_tokenization = list()
        self.special_characters = list()
        
    def print_info_by_attr(self, attribute_name: str):

        print(getattr(self, attribute_name, "Attribute not found"))
        return
    

    def __parse_metadata(self):
        """
            Parses metadata from gutenberg info
        """
        pass

    def get_book(self, url: str, from_txt: bool = False, txt_file_path: str = None):
        """
            get_book: Using given URL, attempt to get text

                Only works with Project Guttenburg since this is a class project

            Parameters:
                url: Url to guttenberg page containing ebook text
            
            return: int: 0 for success, 1 for failure
        """

        if from_txt is True:
            if txt_file_path is None:
                print("get_book(): No file specified! Falling back to URL", file=sys.stderr)
            try:
                with open(txt_file_path, 'r') as f:
                    self.raw_text = f.read()
                    print(f"get_book(): Text obtained for book number {self.book_num}")
                    return 0
            except:
                print("get_book(): Failed to open file! Falling back to URL", file=sys.stderr)

        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            self.raw_text = soup.text
            return 0
        else:
            print("get_book(): Failed to retrieve the webpage", file=sys.stderr)
            return 1

    def tokenize(self):
        
        self.full_text_tokenization = word_tokenize(self.pre_process_string)
        self.full_text_tokenization = [word for word in self.full_text_tokenization if word not in string.punctuation]        

        stop_words = set(stopwords.words('english'))
        self.text_tokenization_no_stop = [word for word in self.full_text_tokenization if word not in stop_words]
        
        # print(self.full_text_tokenization)
        # print(self.text_tokenization_no_stop)

    def __extract_chapters(self):
        
        print(f"Extracting chapters for book {self.book_num}")
        
        """_summary_

            Seperates out chapters in each book. Must specify a book number
                as each book has different chapter regexs going on. 
            
        Returns:
            1 on an error, 0 on success
        """

        # Set prefixes / suffixes for regex patterns
        chapter_positions = []
        chapters_nums = chapter_prefix = chapter_suffix = None

        if self.book_num == 1:
            chapters_nums = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 
                             'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV']
            chapter_prefix = "Chapter "
            chapter_suffix = ".\n"
        elif self.book_num == 2:
            # Fill in when we get a new book
            chapters_nums  = ["The Blue Cross", "The Secret Garden", "The Queer Feet", 
                              "The Flying Stars", "The Invisible Man", "The Honour of Israel Gow", 
                              "The Wrong Shape", "The Sins of Prince Saradine", "The Hammer of God", 
                              "The Eye of Apollo", "The Sign of the Broken Sword", 
                              "The Three Tools of Death"]
            chapter_prefix = "\n"
            chapter_suffix = "\n"
        elif self.book_num == 3:
            chapters_nums = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']
            chapter_prefix = "\n"
            chapter_suffix = r"\..*?\n"            
        else:
            print("Error: Book number invalid!", file=sys.stderr)
            return 1
        
        # Get chapter character positions
        for chap in chapters_nums:
            pattern = f"{chapter_prefix}{chap}{chapter_suffix}"
            search_result = re.search(pattern, self.raw_text, flags=re.DOTALL | re.IGNORECASE)
            chapter_positions.append((search_result.start(), search_result.end()))
        
        # Using chapter character positions, extract each chapter
        for idx in range(len(chapter_positions)):

            if idx == len(chapter_positions) - 1:
                self.chapters.append(self.raw_text[chapter_positions[idx][1]:])
            else:
                self.chapters.append(self.raw_text[chapter_positions[idx][1]: chapter_positions[idx + 1][0]])

        return 0

    def __extract_paragraphs(self):
        
        """_summary_
            Goes through each chapter, and splits chapters into their paragraphs
            
            The paragraph delimiter is '\\n\\n'
            
            self.paragraphs is a 2d list [chapter][paragraph]
            
            self.global_paragraphs: 1d list of ALL paragraphs
                Have not added this in yet. Can though, wouldn't be all that hard
            
        """
        for chapter in self.chapters:
            self.paragraphs.append(chapter.split("\n\n"))
            self.paragraphs[-1] = [x.replace("\n"," ") for x in self.paragraphs[-1]]
            self.paragraphs[-1] = [x for x in self.paragraphs[-1] if x != '']
        
    def __extract_sentences(self):
        
        for chapter in self.paragraphs:
            for paragraph in chapter:
                self.sentences.extend(sent_tokenize(paragraph))

        for i in range(len(self.sentences)):
            self.sentences[i] = ' '.join(self.sentences[i].split())

        # print(*self.sentences, sep="\n")
        
    def __clean_raw_string(self):
        
        self.raw_text = self.raw_text.lower()
        self.raw_text = self.raw_text.replace("\r", "")
        
        # Remove special characters, collect them for now. Probably can get rid
        #   of storing them. 
        for character in self.raw_text:
            if character.isascii() is False:
                self.special_characters.append(character)
                
        self.special_characters = list(set(self.special_characters))
        self.raw_text = unidecode(self.raw_text)
        
        # Strip certain punctuation
        self.raw_text = self.raw_text.replace(";", " ")
        self.raw_text = self.raw_text.replace("-"," ")
        self.raw_text = self.raw_text.replace("—", " ")
        self.raw_text = self.raw_text.replace(",", " ")
        self.raw_text = self.raw_text.replace("(", " ")
        self.raw_text = self.raw_text.replace(")", " ")
        
    def __strip_header_footer(self):
        # Strip header
        pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*? \*\*\*\n"
        search_result = re.search(pattern, self.raw_text, flags=re.DOTALL | re.IGNORECASE)
        
        if search_result:
            self.raw_text = self.raw_text[search_result.end():]
        else:
            print("Warning: Header not found. Proceeding without stripping header.")
        
        # Strip footer
        pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*? \*\*\*\n"
        search_result = re.search(pattern, self.raw_text, flags=re.DOTALL | re.IGNORECASE)
        
        if search_result:
            self.raw_text = self.raw_text[:search_result.start()]
        else:
            print("Warning: Footer not found. Proceeding without stripping footer.")

        
    def __combine_cleaned_sentences(self):
        
        self.pre_process_string = ' '.join(self.sentences)
        # print(self.combined_sentence)
        
    def extract_names(self):
        """
            Using spacy en_core_web_sm model, attempt to extract all names
                from the txt file. We go sentence by sentence in order to 
                get a more accurate extraction. Sentences offer more of 
                a bound.
            
            self.names is set to whatever names are extracted

            Requirements that you download: 
        """
        nlp = spacy.load("en_core_web_sm")
        all_names = []
        for sentence in self.sentences:
            doc = nlp(sentence)
            
            names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

            all_names.extend(names)

        self.names = list(set(all_names))

    def pre_process(self):
                
        self.__strip_header_footer()
        self.__clean_raw_string()
        self.__extract_chapters()
        self.__extract_paragraphs()
        self.__extract_sentences()
        self.__combine_cleaned_sentences()
        self.extract_names()
        
