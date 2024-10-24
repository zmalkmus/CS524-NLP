import re
import requests
import sys
from bs4 import BeautifulSoup

from unidecode import unidecode

import spacy
import string

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords, names
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK packages are downloaded
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('names')

class Book():
    def __init__(self, book_number: int = None, txt_file_path: str = None, is_story: bool = False) -> None:
        """
        Initialize the Book class.

        Parameters:
            book_number (int): Identifier for the book.
            txt_file_path (str): Path to the text file containing the book or story.
            is_story (bool): Indicates if the text is an individual story without chapters.
        """
        print(f"[__init__]: Initializing book: {book_number}")
        self.book_num = book_number
        self.txt_file_path = txt_file_path
        self.is_story = is_story
        self.raw_text = None

        self.chapters = list()  # Holds a string for each of the chapters
        self.paragraphs = list()  # Holds lists of strings for each paragraph in each chapter
        self.sentences = list()
        self.pre_process_string = str()

        self.names = list()
        self.name_variations = dict()
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

        # Initialize stopwords and names
        self.stop_words = set(stopwords.words('english'))
        self.common_words = set(stopwords.words('english')) | set(nltk.corpus.words.words())
        self.all_names = set(names.words())

    def print_info_by_attr(self, attribute_name: str):
        print(getattr(self, attribute_name, "Attribute not found"))
        return

    def get_book(self, url: str = None, from_txt: bool = False, txt_file_path: str = None):
        """
        Get the book text either from a URL or a local text file.

        Parameters:
            url (str): URL to the Gutenberg page containing ebook text.
            from_txt (bool): If True, read the book from a local text file.
            txt_file_path (str): Path to the text file containing the book.

        Returns:
            int: 0 for success, 1 for failure
        """
        if from_txt:
            if txt_file_path is None and self.txt_file_path is None:
                print("get_book(): No file specified!", file=sys.stderr)
                return 1
            file_path = txt_file_path if txt_file_path else self.txt_file_path
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.raw_text = f.read()
                    print(f"get_book(): Text obtained from file {file_path}")
                    return 0
            except Exception as e:
                print(f"get_book(): Failed to open file! Error: {e}", file=sys.stderr)
                return 1

        elif url:
            response = requests.get(url)
            if response.status_code == 200:
                self.raw_text = response.text
                print(f"get_book(): Text obtained from URL {url}")
                return 0
            else:
                print("get_book(): Failed to retrieve the webpage", file=sys.stderr)
                return 1
        else:
            print("get_book(): No source specified for the book text.", file=sys.stderr)
            return 1

    def tokenize(self):
        """
        Tokenizes the entire text string, as well as the sentences.

        We also strip any remaining punctuation and get rid of stop words.
        """
        # Convert text to lowercase for tokenization
        lower_text = self.pre_process_string.lower()
        self.text_tokenized = word_tokenize(lower_text)

        # Full text string
        self.text_tokenized = [word.strip("\n") for word in self.text_tokenized if
                               word.isalpha() and word not in self.stop_words]

        # Sentences:
        for sentence in self.sentences:
            sentence_lower = sentence.lower()
            tokens = word_tokenize(sentence_lower)
            tokens = [word.strip("\n") for word in tokens if word.isalpha() and word not in self.stop_words]
            sentence_tokenized = ' '.join(tokens)
            if sentence_tokenized != '':
                self.sentences_tokenized.append(sentence_tokenized)

    def __extract_chapters(self):
        """
        Separates out chapters in each book. Must specify a book number
        as each book has different chapter regexes.

        If the book is an individual story (`is_story=True`), the entire
        text is treated as one chapter.
        """
        print(f"Extracting chapters for book {self.book_num}")

        if self.is_story:
            # Treat the entire text as one chapter
            self.chapters.append(self.raw_text)
            lower_chapter = self.raw_text.lower()
            tokens = word_tokenize(lower_chapter)
            tokens = [word.strip("\n") for word in tokens if word.isalpha() and word not in self.stop_words]
            chapter_normalized = ' '.join(tokens)
            self.chapters_normalized.append(chapter_normalized)
            return 0

        # Set prefixes / suffixes for regex patterns
        chapter_positions = []
        chapters_nums = chapter_prefix = chapter_suffix = None

        if self.book_num == 1:
            chapters_nums = [r'I', r'II', r'III', r'IV', r'V', r'VI', r'VII', r'VIII',
                             r'IX', r'X', r'XI', r'XII', r'XIII', r'XIV', r'XV']
            chapter_prefix = r"Chapter "
            chapter_suffix = r".\n.*?\n"  # Gets rid of the next line (chapter title)
        elif self.book_num == 2:
            chapters_nums = [r"The Blue Cross", r"The Secret Garden", r"The Queer Feet",
                             r"The Flying Stars", r"The Invisible Man", r"The Honour of Israel Gow",
                             r"The Wrong Shape", r"The Sins of Prince Saradine", r"The Hammer of God",
                             r"The Eye of Apollo", r"The Sign of the Broken Sword",
                             r"The Three Tools of Death"]
            chapter_prefix = r"\n"
            chapter_suffix = r"\n"
        elif self.book_num == 3:
            chapters_nums = [r'I', r'II', r'III', r'IV', r'V', r'VI', r'VII', r'VIII']
            chapter_prefix = r"\n"
            chapter_suffix = r"\..*?\n"
        else:
            # For other books or stories without predefined chapters
            print("No chapter extraction rules defined for this book. Treating entire text as one chapter.")
            self.chapters.append(self.raw_text)
            lower_chapter = self.raw_text.lower()
            tokens = word_tokenize(lower_chapter)
            tokens = [word.strip("\n") for word in tokens if word.isalpha() and word not in self.stop_words]
            chapter_normalized = ' '.join(tokens)
            self.chapters_normalized.append(chapter_normalized)
            return 0

        # Get chapter character positions
        for chap in chapters_nums:
            pattern = f"{chapter_prefix}{chap}{chapter_suffix}"
            search_result = re.search(pattern, self.raw_text, flags=re.DOTALL | re.IGNORECASE)
            if search_result:
                chapter_positions.append((search_result.start(), search_result.end()))
            else:
                print(f"Warning: Chapter {chap} not found.")

        if not chapter_positions:
            print("No chapters found. Treating entire text as one chapter.")
            self.chapters.append(self.raw_text)
            lower_chapter = self.raw_text.lower()
            tokens = word_tokenize(lower_chapter)
            tokens = [word.strip("\n") for word in tokens if word.isalpha() and word not in self.stop_words]
            chapter_normalized = ' '.join(tokens)
            self.chapters_normalized.append(chapter_normalized)
            return 0

        # Using chapter character positions, extract each chapter
        for idx in range(len(chapter_positions)):

            if idx == len(chapter_positions) - 1:
                chapter_text = self.raw_text[chapter_positions[idx][1]:]
            else:
                chapter_text = self.raw_text[chapter_positions[idx][1]: chapter_positions[idx + 1][0]]

            self.chapters.append(chapter_text)

            # Normalize chapter text for word counts, etc.
            lower_chapter = chapter_text.lower()
            tokens = word_tokenize(lower_chapter)
            tokens = [word.strip("\n") for word in tokens if word.isalpha() and word not in self.stop_words]
            chapter_normalized = ' '.join(tokens)
            self.chapters_normalized.append(chapter_normalized)

        return 0

    def __extract_paragraphs(self):

        """_summary_
            Goes through each chapter, and splits chapters into their paragraphs

            The paragraph delimiter is '\\n\\n'

            self.paragraphs is a 2D list [chapter][paragraph]
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
        for i in range(len(self.sentences)):
            self.sentences[i] = ' '.join(self.sentences[i].split())

    def __clean_raw_string(self):
        """
            Clean the raw text by removing unwanted characters and normalizing.
        """
        # Remove '\r', since Gutenberg uses it
        self.raw_text = self.raw_text.replace("\r", "")

        # Remove special characters, collect them for now
        self.raw_text = unidecode(self.raw_text)

        # Remove unwanted punctuation
        for punct in [';', '-', '—', ',', '(', ')', '`']:
            self.raw_text = self.raw_text.replace(punct, ' ')

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

    def extract_names(self):
        """
        Improved name extraction method that filters out non-names and
        maps different mentions of the same character to a canonical name.
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.pre_process_string)
        all_names = []

        # Extract PERSON entities
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                all_names.append(ent.text)

        # Remove any extra whitespace
        all_names = [name.strip() for name in all_names if name.strip()]

        # Build a frequency distribution
        name_freq = nltk.FreqDist(all_names)

        # Exclude names that are common words or too short
        common_words = self.common_words
        titles = set(['Mr', 'Mrs', 'Miss', 'Sir', 'Lady', 'Dr', 'Master', 'Captain', 'Uncle', 'Aunt'])
        filtered_names = []
        for name in all_names:
            parts = name.split()
            if any(part.lower() in common_words or part in titles or len(part) <= 2 for part in parts):
                continue
            filtered_names.append(name)

        # Map name variations to canonical names
        name_variations = {}
        canonical_names = {}

        for name in filtered_names:
            parts = name.split()
            canonical_name = name  # Assume the full name is canonical
            for part in parts:
                if part not in common_words and len(part) > 2:
                    name_variations[part] = canonical_name
            name_variations[name] = canonical_name
            canonical_names[canonical_name] = canonical_names.get(canonical_name, 0) + name_freq[name]

        # Keep only the most frequent canonical names
        min_freq = 2  # Adjust this threshold as needed
        self.names = [name for name, freq in canonical_names.items() if freq >= min_freq]
        self.name_variations = name_variations

    # ========================= Feature Extraction =========================

    def __get_word_count(self):
        """
            Get the word count for the book
        """
        self.word_count = len(self.text_tokenized)

    def __get_sentence_count(self):
        """
            Get the sentence count for the book
        """
        self.sentence_count = len(self.sentences_tokenized)

    def __get_chapter_count(self):
        """
            Get the chapter count for the book
        """
        self.chapter_count = len(self.chapters_normalized)

    # ========================= Character Features =========================

    def __extract_first_mentions(self):
        """
            Find the first mention of each name in the book by chapter
        """
        for name in self.names:
            self.character_mentions_first[name] = -1
            for idx, chapter in enumerate(self.chapters):
                if name in chapter:
                    self.character_mentions_first[name] = idx
                    break

    def __extract_all_mentions(self):
        """
        Extract the total number of mentions of each name in the book.
        """
        for name in self.names:
            count = self.pre_process_string.count(name)
            self.character_mentions_all[name] = count

    def __extract_character_proximity(self):
        """
        Extracts the proximity of characters based on surrounding sentences.
        """
        if not hasattr(self, 'nlp'):
            self.nlp = spacy.load("en_core_web_sm")

        # Process sentences to extract names in each sentence
        names_in_sentence = []
        for sentence in self.sentences:
            doc = self.nlp(sentence)
            names = set()
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    name = ent.text.strip()
                    if name in self.name_variations:
                        canonical_name = self.name_variations[name]
                        names.add(canonical_name)
            names_in_sentence.append(names)

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
                    key = tuple(sorted((name1, name2)))

                    # Increment the count for this pair
                    self.character_proximity[key] = self.character_proximity.get(key, 0) + 1

    # ========================= Event Features =========================

    def pre_process(self):

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
