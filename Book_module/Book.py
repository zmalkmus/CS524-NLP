import sys
import re
from collections import defaultdict, Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, names
from nltk.sentiment import SentimentIntensityAnalyzer

import spacy
from unidecode import unidecode

nltk.download('punkt', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('names', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class Book:

    def __init__(self, file_path: str):
        """
        file_path: Path to the book text file to process
        """
        print(f"[__init__]: Initializing book from file: {file_path}")
        self.file_path = file_path
        self.raw_text = None
        self.normalized_text = None
        
        self.chapters = []  # Holds a string for each of the chapters
        self.paragraphs = []  # Holds lists of strings for each paragraph in each chapter
        self.sentences = []
        self.pre_process_string = ""
        
        self.names = []
        self.name_variations = {}
        self.names_tokenized = []
        self.special_characters = []

        # These should be the final tokenized / cleaned text
        self.sentences_tokenized = []
        self.text_tokenized = []
        self.chapters_normalized = []

        # Features
        self.word_count = 0
        self.sentence_count = 0
        self.character_mentions_all = {}
        self.character_proximity = {}
        self.character_mentions_first = {}
        self.chapter_count = 0

        self.stop_words = set(stopwords.words('english'))
        self.common_words = set(stopwords.words('english')) | set(nltk.corpus.words.words())
        self.all_names = set(names.words())

        # New features
        self.character_sentiments = {}
        self.crime_keywords = [
            'murder', 'kill', 'dead', 'body', 'weapon', 'crime', 'blood', 'knife', 
            'gun', 'death', 'victim', 'suspect'
        ]
        self.crime_keyword_frequency = Counter()
        self.crime_first_introduction = -1
            
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

    def normalize(self):
        """
        Normalizes the raw text by:
        - Stripping headers and footers
        - Removing special characters
        - Replacing non-ASCII characters
        - Stripping punctuation
        - Assigning the normalized text to self.normalized_text
        """
        # Strip Project Gutenberg header and footer if present
        # Header
        pattern_header = r"\*\*\* START OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*\n"
        search_result = re.search(pattern_header, self.raw_text, flags=re.DOTALL | re.IGNORECASE)
        if search_result:
            self.raw_text = self.raw_text[search_result.end():]
        else:
            print("Warning: Start of Project Gutenberg header not found.", file=sys.stderr)
        
        # Footer
        pattern_footer = r"\*\*\* END OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*\n"
        search_result = re.search(pattern_footer, self.raw_text, flags=re.DOTALL | re.IGNORECASE)
        if search_result:
            self.raw_text = self.raw_text[:search_result.start()]
        else:
            print("Warning: End of Project Gutenberg footer not found.", file=sys.stderr)
        
        # Remove carriage returns
        text = self.raw_text.replace('\r', '')

        # Remove special characters, collect them
        text = unidecode(text)
        self.special_characters = [char for char in set(text) if not char.isascii()]
        
        # Strip certain punctuation
        punctuation_to_remove = [";", "-", "—", ",", "(", ")", "`", "'", '"']
        for p in punctuation_to_remove:
            text = text.replace(p, " ")

        self.normalized_text = text

    def tokenize(self):
        """
        Tokenizes the entire text string and sentences.
        Converts tokens to lowercase and removes stop words and non-alphabetic tokens.
        """
        tokens = word_tokenize(self.pre_process_string)
        
        self.text_tokenized = [
            word.lower() for word in tokens 
            if word.isalpha() and word.lower() not in self.stop_words
        ]

        for sentence in self.sentences:
            tokens = word_tokenize(sentence)
            tokenized_sentence = [
                word.lower() for word in tokens 
                if word.isalpha() and word.lower() not in self.stop_words
            ]
            sentence_str = ' '.join(tokenized_sentence)
            if sentence_str:
                self.sentences_tokenized.append(sentence_str)
                            
    def __extract_chapters(self):
        """
        Separates out chapters in the book using common chapter indicators.

        Returns:
            int: 0 on success, 1 on error
        """
        print("Extracting chapters from the book.")
        # Common patterns for chapters
        chapter_patterns = [
            r"^chapter\s+\d+",          # Matches 'Chapter 1', 'Chapter 2', etc.
            r"^chapter\s+[ivxlcdm]+",   # Matches 'Chapter I', 'Chapter II', etc.
            r"^\d+\.\s+",               # Matches '1. ', '2. ', etc.
            r"^[ivxlcdm]+\.\s+",        # Matches 'I. ', 'II. ', etc.
            r"^the\s+.+",               # Matches 'The Beginning', 'The Journey', etc.
        ]

        # Normalize the text for consistent chapter detection
        lines = self.normalized_text.split('\n')
        chapter_indices = []
        for idx, line in enumerate(lines):
            normalized_line = line.strip()
            for pattern in chapter_patterns:
                if re.match(pattern, normalized_line, re.IGNORECASE):
                    chapter_indices.append(idx)
                    break

        # If no chapters are found, treat the entire text as one chapter
        if not chapter_indices:
            self.chapters.append(self.normalized_text)
            self.chapters_normalized.append(self.normalized_text)
            return 0

        # Extract chapters based on identified indices
        for i in range(len(chapter_indices)):
            start_idx = chapter_indices[i]
            end_idx = chapter_indices[i + 1] if i + 1 < len(chapter_indices) else len(lines)
            chapter = '\n'.join(lines[start_idx:end_idx])
            self.chapters.append(chapter)

            # Normalize chapter text
            chapter_normalized = ' '.join([
                word.strip("\n") for word in word_tokenize(chapter)
                if word.isalpha() and word.lower() not in self.stop_words
            ])
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
        """
        Tokenizes paragraphs into sentences and cleans up extra spaces.
        """
        for chapter in self.paragraphs:
            for paragraph in chapter:
                self.sentences.extend(sent_tokenize(paragraph))

        # Strip any extra spaces.
        self.sentences = [' '.join(sentence.split()) for sentence in self.sentences]
            
    def __combine_cleaned_sentences(self):
        """
        Combines cleaned sentences into a single string for further processing.
        """
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
        titles = set(['Mr', 'Mrs', 'Miss', 'Sir', 'Lady', 'Dr', 'Master', 'Captain', 'Uncle', 'Aunt'])
        filtered_names = []
        for name in all_names:
            parts = name.split()
            if any(part.lower() in self.common_words or part in titles or len(part) <= 2 for part in parts):
                continue
            filtered_names.append(name)

        # Map name variations to canonical names
        name_variations = {}
        canonical_names = {}

        for name in filtered_names:
            parts = name.split()
            canonical_name = name  # Assume the full name is canonical
            for part in parts:
                if part not in self.common_words and len(part) > 2:
                    name_variations[part] = canonical_name
            name_variations[name] = canonical_name
            canonical_names[canonical_name] = canonical_names.get(canonical_name, 0) + name_freq[name]

        # Keep only the most frequent canonical names
        min_freq = 2  # Adjust this threshold as needed
        self.names = [name for name, freq in canonical_names.items() if freq >= min_freq]
        self.name_variations = name_variations

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
        Find the first mention of each name in the book by word index.
        """
        # Ensure that both text tokens and name tokens are in lowercase
        for name in self.names:
            name_tokens = [token.lower() for token in name.split()]
            name_length = len(name_tokens)
            found = False
            for idx in range(len(self.text_tokenized) - name_length + 1):
                if self.text_tokenized[idx:idx + name_length] == name_tokens:
                    self.character_mentions_first[name] = idx
                    found = True
                    break
            if not found:
                self.character_mentions_first[name] = -1

    def __extract_all_mentions(self):
        """
        Extract the total number of mentions of each name in the book.
        """
        for name in self.names:
            name_tokens = [token.lower() for token in name.split()]
            count = 0
            for idx in range(len(self.text_tokenized) - len(name_tokens) + 1):
                if self.text_tokenized[idx:idx+len(name_tokens)] == name_tokens:
                    count += 1
            self.character_mentions_all[name] = count

    def __extract_character_proximity(self):
        """
        Extracts the proximity of characters based on surrounding sentences.
        """
        nlp = spacy.load("en_core_web_sm")

        # Process sentences to extract names in each sentence
        names_in_sentence = []
        for sentence in self.sentences:
            doc = nlp(sentence)
            # Get the set of names in the sentence
            sentence_names = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
            names_in_sentence.append(sentence_names)

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

    # ========================= New Feature Extraction Methods =========================

    def __character_sentiment_analysis(self):
        """
        Performs sentiment analysis on sentences related to each character.
        """
        sia = SentimentIntensityAnalyzer()
        for character in self.names:
            sentiments = []
            for sentence in self.sentences:
                if character in sentence:
                    sentiment = sia.polarity_scores(sentence)['compound']
                    sentiments.append(sentiment)
            if sentiments:
                average_sentiment = sum(sentiments) / len(sentiments)
                self.character_sentiments[character] = average_sentiment
            else:
                self.character_sentiments[character] = 0

    def __analyze_crime_keywords(self):
        """
        Analyzes frequency and distribution of crime-related keywords.
        """
        for sentence in self.sentences:
            tokens = word_tokenize(sentence.lower())
            for keyword in self.crime_keywords:
                if keyword in tokens:
                    self.crime_keyword_frequency[keyword] += 1

    def __find_crime_first_introduction(self):
        """
        Determines the position in the text where crime is first introduced.
        """
        for idx, sentence in enumerate(self.sentences):
            tokens = word_tokenize(sentence.lower())
            if any(keyword in tokens for keyword in self.crime_keywords):
                self.crime_first_introduction = idx
                break
        else:
            self.crime_first_introduction = -1  # Indicates not found

    # ========================= Event Features =========================

    def pre_process(self):
        if self.__get_book() == 0:
            self.normalize()
            self.__extract_chapters()
            self.__extract_paragraphs()
            self.__extract_sentences()
            self.__combine_cleaned_sentences()
            self.extract_names()
            self.tokenize()
        else:
            print("Error: Could not get the book text.", file=sys.stderr)

    def feature_extraction(self):
        self.__extract_first_mentions()
        self.__extract_character_proximity()
        self.__extract_all_mentions()
        self.__get_chapter_count()
        self.__get_sentence_count()
        self.__get_word_count()
        self.__character_sentiment_analysis()
        self.__analyze_crime_keywords()
        self.__find_crime_first_introduction()
