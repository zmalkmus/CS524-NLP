import sys
import re
from collections import Counter

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
        self.raw_text = ""
        self.normalized_text = ""
        self.text_tokenized = []
        self.sentences = []
        self.special_characters = []
        self.names = []
        self.name_variations = {}
        self.character_mentions_first = {}
        self.character_mentions_all = {}
        self.character_proximity = {}
        self.character_sentiments = {}
        self.crime_keyword_frequency = Counter()
        self.crime_first_introduction = -1
        self.events = []
        # Counts
        self.word_count = 0
        self.sentence_count = 0
        self.chapter_count = 0
        # Resources
        self.stop_words = set(stopwords.words('english'))
        self.common_words = set(stopwords.words('english')) | set(nltk.corpus.words.words())
        self.all_names = set(names.words())
        self.crime_keywords = [
            'murder', 'kill', 'dead', 'body', 'weapon', 'crime', 'blood', 'knife', 
            'gun', 'death', 'victim', 'suspect'
        ]

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

    def __normalize(self):
        """
        Normalizes the raw text by:
        - Stripping headers and footers
        - Removing special characters
        - Replacing non-ASCII characters
        - Stripping punctuation
        """
        text = self.raw_text
        # Strip Project Gutenberg header and footer if present
        # Header
        pattern_header = r"\*\*\* START OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*\n"
        search_result = re.search(pattern_header, text, flags=re.DOTALL | re.IGNORECASE)
        if search_result:
            text = text[search_result.end():]
        else:
            print("Warning: Start of Project Gutenberg header not found.", file=sys.stderr)
        
        # Footer
        pattern_footer = r"\*\*\* END OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*\n"
        search_result = re.search(pattern_footer, text, flags=re.DOTALL | re.IGNORECASE)
        if search_result:
            text = text[:search_result.start()]
        else:
            print("Warning: End of Project Gutenberg footer not found.", file=sys.stderr)
        
        # Remove carriage returns
        text = text.replace('\r', '')
        # Remove special characters, collect them
        text = unidecode(text)
        self.special_characters = [char for char in set(text) if not char.isascii()]
        # Strip certain punctuation
        punctuation_to_remove = [";", "-", "—", ",", "(", ")", "`", "'", '"']
        for p in punctuation_to_remove:
            text = text.replace(p, " ")
        # Save the cleaned text
        self.normalized_text = text

    def __tokenize(self):
        """
        Tokenizes the cleaned text into words and sentences.
        """
        self.normalized_text = self.normalized_text.lower()
        # Tokenize into sentences
        self.sentences = sent_tokenize(self.normalized_text)
        # Clean up extra spaces in sentences
        self.sentences = [' '.join(sentence.split()) for sentence in self.sentences]
        # Tokenize into words
        tokens = word_tokenize(self.normalized_text)
        self.text_tokenized = [
            word.lower() for word in tokens 
            if word.isalpha() and word.lower() not in self.stop_words
        ]

    def __extract_names(self):
        """
        Improved name extraction method that filters out non-names and
        maps different mentions of the same character to a canonical name.
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.normalized_text)
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
        self.sentence_count = len(self.sentences)
    
    def __get_chapter_count(self):
        """Get the chapter count for the book"""
        # Assuming chapters are defined by "Chapter" headings
        chapter_pattern = re.compile(r'^Chapter', re.IGNORECASE | re.MULTILINE)
        self.chapter_count = len(chapter_pattern.findall(self.normalized_text))
    
    # ========================= Character Features =========================

    def __extract_first_mentions(self):
        """
        Find the first mention of each name in the book by word index.
        """
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

    def __get_subjects(self, token):
        subjects = []
        for child in token.children:
            if child.dep_ in ('nsubj', 'nsubjpass'):
                subjects.append(child.text)
            elif child.dep_ in ('ccomp', 'xcomp'):
                subjects.extend(self.__get_subjects(child))
        return subjects

    def __get_objects(self, token):
        objects = []
        for child in token.children:
            if child.dep_ in ('dobj', 'pobj', 'obj', 'dative', 'attr', 'oprd'):
                objects.append(child.text)
            elif child.dep_ in ('ccomp', 'xcomp'):
                objects.extend(self.__get_objects(child))
        return objects

    def __extract_events(self):
        nlp = spacy.load('en_core_web_sm')
        self.events = []

        # Define relevant verbs and event types
        event_verbs = {
            # Crime Occurrence
            'murder': 'crime_occurrence',
            'kill': 'crime_occurrence',
            'poison': 'crime_occurrence',
            'assassinate': 'crime_occurrence',
            'slay': 'crime_occurrence',
            'shoot': 'crime_occurrence',
            'stab': 'crime_occurrence',
            'strangle': 'crime_occurrence',

            # Discovery and Investigation
            'discover': 'discovery',
            'find': 'discovery',
            'uncover': 'discovery',
            'investigate': 'investigation',
            'search': 'investigation',
            'probe': 'investigation',
            'examine': 'investigation',
            'inspect': 'investigation',
            'interrogate': 'investigation',

            # Legal Actions
            'arrest': 'legal_action',
            'charge': 'legal_action',
            'trial': 'legal_action',
            'testify': 'legal_action',
            'convict': 'legal_action',
            'sentence': 'legal_action',

            # Confession and Revelation
            'confess': 'confession',
            'admit': 'confession',
            'reveal': 'revelation',
            'expose': 'revelation',
            'disclose': 'revelation',

            # Suspicion and Accusation
            'suspect': 'suspicion',
            'accuse': 'accusation',
            'blame': 'accusation',
            'denounce': 'accusation',

            # Confrontation and Conflict
            'confront': 'confrontation',
            'fight': 'conflict',
            'argue': 'conflict',
            'challenge': 'confrontation',
            'threaten': 'conflict',
            'attack': 'conflict',

            # Deception and Betrayal
            'betray': 'betrayal',
            'deceive': 'deception',
            'lie': 'deception',
            'disguise': 'deception',
            'manipulate': 'deception',
            'sabotage': 'betrayal',

            # Escape and Evasion
            'escape': 'evasion',
            'flee': 'evasion',
            'run': 'evasion',
            'hide': 'evasion',

            # Rescue and Protection
            'rescue': 'rescue',
            'save': 'rescue',
            'protect': 'protection',
            'guard': 'protection',
            'defend': 'protection',

            # Planning and Strategy
            'plan': 'planning',
            'plot': 'planning',
            'scheme': 'planning',
            'organize': 'planning',

            # Surveillance and Observation
            'observe': 'surveillance',
            'watch': 'surveillance',
            'monitor': 'surveillance',
            'spy': 'surveillance',

            # Emotional and Psychological Actions
            'warn': 'warning',
            'threaten': 'threat',
            'confide': 'trust',
            'fear': 'emotion',

            # Miscellaneous Relevant Actions
            'ambush': 'attack',
            'trap': 'entrapment',
            'pursue': 'pursuit',
            'follow': 'pursuit',
            'question': 'interrogation',
            'gather': 'meeting',
            'assemble': 'meeting',
            'proclaim': 'announcement',
            'announce': 'announcement',
        }

        for idx, sentence in enumerate(self.sentences):
            doc = nlp(sentence)
            for token in doc:
                if token.lemma_ in event_verbs and token.pos_ == 'VERB':
                    event = {
                        'sentence_idx': idx,
                        'sentence': sentence,
                        'verb': token.lemma_,
                        'event_type': event_verbs[token.lemma_],
                        'subject': None,
                        'object': None,
                        'characters': [],
                    }

                    # Extract subjects and objects using helper functions
                    subjects = self.__get_subjects(token)
                    objects = self.__get_objects(token)

                    if subjects:
                        event['subject'] = ', '.join(subjects)
                    if objects:
                        event['object'] = ', '.join(objects)

                    # Identify characters involved
                    characters_in_sentence = [
                        name for name in self.names if name in sentence
                    ]
                    event['characters'] = characters_in_sentence

                    self.events.append(event)

    def pre_process(self):
        if self.__get_book() == 0:
            self.__normalize()
            self.__extract_names() # !!! Extract names here before we make lowercase !!!
            self.__tokenize()
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
        self.__extract_events()
