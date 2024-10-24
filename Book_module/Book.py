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
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class Book:

    def __init__(self, file_path: str):
        """
            file_path: Path to the book text file to process
        """
        print(f"[__init__]: Initializing book from file: {file_path}")
        self.file_path = file_path
        self.raw_text = None
        
        self.chapters = list() # Holds a string for each of the chapters
        self.paragraphs = list() # Holds lists of strings for each paragraph in each chapter
        self.sentences = list()
        self.pre_process_string = str()
        
        self.names = list()
        self.names_tokenized = list()
        self.special_characters = list()

        # These should be the final tokenized / cleaned text
        self.sentences_tokenized = list()
        self.text_tokenized = list()
        self.chapters_normalized = list()

        # Features
        self.word_count = 0
        self.sentence_count = 0
        self.character_mentions_all = dict()
        self.character_proximity = dict()
        self.character_mentions_first = dict()
        self.chapter_count = 0
            
    def print_info_by_attr(self, attribute_name: str):
        print(getattr(self, attribute_name, "Attribute not found"))
        return

    def __get_book(self):
        """
            Reads the book text from the file specified in self.file_path.

            Returns:
                int: 0 for success, 1 for failure
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.raw_text = f.read()
                print(f"get_book(): Text obtained from file '{self.file_path}'.")
                return 0
        except Exception as e:
            print(f"get_book(): Failed to open file '{self.file_path}'. Error: {e}", file=sys.stderr)
            return 1

    def tokenize(self):
        """
            Tokenizes the entire text string, as well as the sentences
            
            We also strip any remaining punctuation, and get rid of stop words
            
            At the end of the sentence tokenization, we call extract names
                since this more than likely would allow for a much better
                name extraction since sentences are cleaner!
        """
        stop_words = set(stopwords.words('english'))
        self.text_tokenized = word_tokenize(self.pre_process_string)
 
        # Full text string
        self.text_tokenized = [word.strip("\n") for word in self.text_tokenized if \
                                word.isalpha() and \
                                word not in stop_words]
        # Sentences:
        for idx in range(len(self.sentences)):
            tokenized_sentence = [word.strip("\n") for word in word_tokenize(self.sentences[idx]) 
                                  if word.isalpha() and word not in stop_words]
            sentence = ' '.join(tokenized_sentence)
            if sentence:
                self.sentences_tokenized.append(sentence)
                            
    def __extract_chapters(self):
        print("Extracting chapters from the book.")
        """
            Separates out chapters in the book using common chapter indicators.

            Returns:
                int: 0 on success, 1 on error
        """
        # Common patterns for chapters
        chapter_patterns = [
            r"^chapter\s+\d+",       # Matches 'Chapter 1', 'Chapter 2', etc.
            r"^chapter\s+[ivx]+",    # Matches 'Chapter I', 'Chapter II', etc.
            r"^\d+\.\s+",            # Matches '1. ', '2. ', etc.
            r"^the\s+.+",            # Matches 'The Beginning', 'The Journey', etc.
        ]

        # Normalize the text for consistent chapter detection
        lines = self.raw_text.split('\n')
        chapter_indices = []
        for idx, line in enumerate(lines):
            normalized_line = line.strip().lower()
            for pattern in chapter_patterns:
                if re.match(pattern, normalized_line):
                    chapter_indices.append(idx)
                    break

        # If no chapters are found, treat the entire text as one chapter
        if not chapter_indices:
            self.chapters.append(self.raw_text)
            self.chapters_normalized.append(self.raw_text)
            return 0

        # Extract chapters based on identified indices
        for i in range(len(chapter_indices)):
            start_idx = chapter_indices[i]
            end_idx = chapter_indices[i + 1] if i + 1 < len(chapter_indices) else len(lines)
            chapter = '\n'.join(lines[start_idx:end_idx])
            self.chapters.append(chapter)

            # Normalize chapter text
            stop_words = set(stopwords.words('english'))
            chapter_normalized = ' '.join([word.strip("\n") for word in word_tokenize(chapter)
                                            if word.isalpha() and word not in stop_words])
            self.chapters_normalized.append(chapter_normalized)

        return 0

    def __extract_paragraphs(self):
        """
            Splits chapters into paragraphs using double newline delimiters.

            self.paragraphs is a list of lists: [chapter][paragraph]
        """
        for chapter in self.chapters:
            paragraphs = chapter.split("\n\n")
            paragraphs = [x.replace("\n", " ") for x in paragraphs]
            paragraphs = [x for x in paragraphs if x.strip() != '']
            self.paragraphs.append(paragraphs)
        
    def __extract_sentences(self):
        # Get full list of sentences
        for chapter in self.paragraphs:
            for paragraph in chapter:
                self.sentences.extend(sent_tokenize(paragraph))

        # Strip any extra spaces.
        self.sentences = [' '.join(sentence.split()) for sentence in self.sentences]
            
    def __clean_raw_string(self):
        """
            Cleans the raw text by:
            - Converting to lowercase
            - Removing carriage returns
            - Replacing non-ASCII characters
            - Stripping certain punctuation
        """
        self.raw_text = self.raw_text.lower()
        self.raw_text = self.raw_text.replace("\r", "")
        
        # Remove special characters, collect them for now.
        for character in self.raw_text:
            if not character.isascii():
                self.special_characters.append(character)
                
        self.special_characters = list(set(self.special_characters))
        
        self.raw_text = unidecode(self.raw_text)
        
        # Strip certain punctuation
        punctuation_to_remove = [";", "-", "—", ",", "(", ")", "`"]
        for p in punctuation_to_remove:
            self.raw_text = self.raw_text.replace(p, " ")
            
    def __strip_header_footer(self):
        # Strip Project Gutenberg header and footer if present
        # Header
        pattern = r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*\n"
        search_result = re.search(pattern, self.raw_text, flags=re.DOTALL | re.IGNORECASE)
        if search_result:
            self.raw_text = self.raw_text[search_result.end():]
        else:
            print("Warning: Start of Project Gutenberg header not found.", file=sys.stderr)
        
        # Footer
        pattern = r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*\n"
        search_result = re.search(pattern, self.raw_text, flags=re.DOTALL | re.IGNORECASE)
        if search_result:
            self.raw_text = self.raw_text[:search_result.start()]
        else:
            print("Warning: End of Project Gutenberg footer not found.", file=sys.stderr)
            
    def __combine_cleaned_sentences(self):
        self.pre_process_string = ' '.join(self.sentences)
            
    def extract_names(self):
        """
            Uses spaCy's model to extract names from the text.
            Processes sentence by sentence for accuracy.
        """
        nlp = spacy.load("en_core_web_sm")
        all_names = []
        for sentence in self.sentences:
            doc = nlp(sentence)
            names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            all_names.extend(names)

        self.names = list(set(all_names))

    # ========================= Feature Extraction =========================

    def __get_word_count(self):
        """Get the word count for the book"""
        self.word_count = len(self.text_tokenized)

    def __get_sentence_count(self):
        """Get the sentence count for the book"""
        self.sentence_count = len(self.sentences_tokenized)
    
    def __get_chapter_count(self):
        """Get the chapter count for the book"""
        self.chapter_count = len(self.chapters_normalized)
    
    # ========================= Character Features =========================

    def __extract_first_mentions(self):
        """
            Find the first mention of each name in the book by chapter
        """
        for name in self.names:
            self.character_mentions_first[name] = -1
            for idx, chapter in enumerate(self.chapters_normalized):
                if name in chapter:
                    self.character_mentions_first[name] = idx
                    break

    def __extract_all_mentions(self):
        """
            Extract the total number of mentions of each name in the book.
        """
        for name in self.names:
            self.character_mentions_all[name] = self.text_tokenized.count(name)

    def __extract_character_proximity(self):
        """
            Extracts the proximity of characters based on surrounding sentences.

            Surrounding Sentences: Expanding to the neighboring sentences 
            (previous and next) could capture indirect interactions, 
            such as when characters are discussed in sequence.
        """
        nlp = spacy.load("en_core_web_sm")

        # Process sentences to extract names in each sentence
        names_in_sentence = []
        for sentence in self.sentences_tokenized:
            doc = nlp(sentence)
            # Get the set of names in the sentence
            names_in_sentence.append(set([ent.text for ent in doc.ents if ent.label_ == "PERSON"]))

        # Surrounding Sentences Proximity
        for i in range(len(names_in_sentence)):
            # Initialize a set to hold names in the surrounding window
            names_in_window = set()
            
            # Include names from the previous sentence if it exists
            if i > 0:
                names_in_window.update(names_in_sentence[i - 1])
            
            # Include names from the current sentence
            names_in_window.update(names_in_sentence[i])
            
            # Include names from the next sentence if it exists
            if i < len(names_in_sentence) - 1:
                names_in_window.update(names_in_sentence[i + 1])
            
            # Generate all unique pairs of names in the window
            names_in_window = list(names_in_window)
            for idx1 in range(len(names_in_window)):
                for idx2 in range(idx1 + 1, len(names_in_window)):
                    name1 = names_in_window[idx1]
                    name2 = names_in_window[idx2]
                    # Create a sorted tuple to avoid duplicate pairs
                    key = tuple(sorted((name1, name2)))

                    # Increment the count for this pair
                    self.character_proximity[key] = self.character_proximity.get(key, 0) + 1

    # ========================= Event Features =========================

    def pre_process(self):
        self.__get_book()
        self.__strip_header_footer()
        self.__clean_raw_string()
        self.__extract_chapters()
        self.__extract_paragraphs()
        self.__extract_sentences()
        self.__combine_cleaned_sentences()
        self.extract_names()
        self.tokenize()

    def feature_extraction(self):
        self.__extract_first_mentions()
        self.__extract_character_proximity()
        self.__extract_all_mentions()
        self.__get_chapter_count()
        self.__get_sentence_count()
        self.__get_word_count()
