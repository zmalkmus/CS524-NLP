import sys
import re
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, names, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

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
        #print(f"[__init__]: Initializing book from file: {file_path}")
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
        self.plot_structure = {}
        
        self.word_count = 0
        self.sentence_count = 0
        self.chapter_count = 0
        
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
                #print(f"get_book(): Text obtained from file '{self.file_path}'.")
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
        #else:
            #print("Warning: Start of Project Gutenberg header not found.", file=sys.stderr)

        # Footer
        pattern_footer = r"\*\*\* END OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*\n"
        search_result = re.search(pattern_footer, text, flags=re.DOTALL | re.IGNORECASE)
        if search_result:
            text = text[:search_result.start()]
        #else:
            #print("Warning: End of Project Gutenberg footer not found.", file=sys.stderr)

        text = text.replace('\r', '')
        text = text.replace('\n', ' ')
        text = text.replace('“', '"')
        text = text.replace('”', '"')

        text = unidecode(text)
        self.special_characters = [char for char in set(text) if not char.isascii()]
        for char in self.special_characters:
            text = text.replace(char, '')

        punctuation_to_remove = [";", "-", "—", ",", "(", ")", "`", '"', "'s", "'"]
        for p in punctuation_to_remove:
            text = text.replace(p, "")

        self.normalized_text = text

    def __tokenize(self):
        """
        Tokenizes the cleaned text into words and sentences.
        Performs lemmatization on both the sentences and word tokens, excluding names.
        """
        self.sentences = sent_tokenize(self.normalized_text)
        lemmatizer = WordNetLemmatizer()

        def get_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN

        lemmatized_sentences = []
        for sentence in self.sentences:
            tokens = word_tokenize(sentence)
            
            pos_tags = nltk.pos_tag(tokens)

            lemmatized_sentence = []
            for token, tag in pos_tags:
                if tag in ('NNP', 'NNPS'):  # Skip proper nouns
                    lemmatized_sentence.append(token)
                else:
                    lemmatized_token = lemmatizer.lemmatize(token, get_pos(tag))
                    lemmatized_sentence.append(lemmatized_token)

            lemmatized_sentences.append(' '.join(lemmatized_sentence))

        self.sentences = lemmatized_sentences

        tokens = word_tokenize(' '.join(self.sentences))
        pos_tags = nltk.pos_tag(tokens)

        lemmatized_tokens = []
        for token, tag in pos_tags:
            if tag in ('NNP', 'NNPS'):
                lemmatized_tokens.append((token, tag))
            else:
                lemmatized_token = lemmatizer.lemmatize(token, get_pos(tag))
                lemmatized_tokens.append((lemmatized_token, tag))

        self.text_tokenized = []
        for word, tag in lemmatized_tokens:
            if word.isalpha():
                if tag not in ('NNP', 'NNPS'):
                    word_lower = word.lower()
                    if word_lower not in self.stop_words:
                        self.text_tokenized.append(word_lower)
                else:
                    self.text_tokenized.append(word)


    def __extract_names(self):
        """
        Improved name extraction method that filters out non-names and
        maps different mentions of the same character to a canonical name.
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.normalized_text)
        all_names = []

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                all_names.append(ent.text)

        all_names = [name.strip() for name in all_names if name.strip()]

        name_freq = nltk.FreqDist(all_names)

        titles = set(['Mr', 'Mrs', 'Miss', 'Sir', 'Lady', 'Dr', 'Master', 'Captain', 'Uncle', 'Aunt'])
        filtered_names = []
        for name in all_names:
            parts = name.split()
            if any(part.lower() in self.common_words or part in titles or len(part) <= 2 for part in parts):
                continue
            filtered_names.append(name)

        name_variations = {}
        canonical_names = {}

        for name in filtered_names:
            parts = name.split()
            canonical_name = name
            for part in parts:
                if part not in self.common_words and len(part) > 2:
                    name_variations[part] = canonical_name
            name_variations[name] = canonical_name
            canonical_names[canonical_name] = canonical_names.get(canonical_name, 0) + name_freq[name]

        min_freq = 2
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
        chapter_pattern = re.compile(r'^Chapter', re.IGNORECASE | re.MULTILINE)
        self.chapter_count = len(chapter_pattern.findall(self.normalized_text))

    # ========================= Character Features =========================

    def __extract_first_mentions(self):
        """
        Find the first mention of each name in the book by word index.
        """
        for name in self.names:
            name_tokens = [token for token in name.split()]
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
            name_tokens = [token for token in name.split()]
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

        names_in_sentence = []
        for sentence in self.sentences:
            doc = nlp(sentence)
            sentence_names = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
            names_in_sentence.append(sentence_names)

        for i in range(len(names_in_sentence)):
            names_in_window = set()

            if i > 0:
                names_in_window.update(names_in_sentence[i - 1])

            names_in_window.update(names_in_sentence[i])

            if i < len(names_in_sentence) - 1:
                names_in_window.update(names_in_sentence[i + 1])

            # Generate all unique pairs of names in the window
            names_in_window = list(names_in_window)
            for idx1 in range(len(names_in_window)):
                for idx2 in range(idx1 + 1, len(names_in_window)):
                    name1 = names_in_window[idx1]
                    name2 = names_in_window[idx2]
                    key = tuple(sorted((name1, name2)))

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
            self.crime_first_introduction = -1

    
    # ========================= Plot Features =========================

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
        event_verbs = get_event_map()

        names_in_sentence = []
        for sentence in self.sentences:
            sentence_names = set()
            for name in self.names:
                if name in sentence:
                    sentence_names.add(name)
            names_in_sentence.append(sentence_names)

        for idx, sentence in enumerate(self.sentences):
            doc = nlp(sentence)
            for token in doc:
                if token.lemma_ in event_verbs and token.pos_ == 'VERB':
                    event = {
                        'sentence_idx': idx,
                        'sentence': sentence.lower(),
                        'verb': token.lemma_,
                        'event_type': event_verbs[token.lemma_],
                        'subject': None,
                        'object': None,
                        'characters': [],
                    }

                    subjects = self.__get_subjects(token)
                    objects = self.__get_objects(token)

                    if subjects:
                        event['subject'] = ', '.join(subjects)
                    if objects:
                        event['object'] = ', '.join(objects)

                    characters_in_event = set()

                    characters_in_event.update(names_in_sentence[idx])

                    if idx > 0:
                        characters_in_event.update(names_in_sentence[idx - 1])
                    if idx > 1:
                        characters_in_event.update(names_in_sentence[idx - 2])
                    if idx < len(names_in_sentence) - 1:
                        characters_in_event.update(names_in_sentence[idx + 1])
                    if idx < len(names_in_sentence) - 2:
                        characters_in_event.update(names_in_sentence[idx + 2])

                    event['characters'] = list(characters_in_event)

                    self.events.append(event)

    def __identify_plot_structure(self):
        """
        Identifies the plot structure of the narrative based on sentiment analysis
        and event density.

        The plot is divided into the following components:
        - Exposition
        - Rising Action
        - Climax
        - Falling Action
        - Resolution
        """

        import numpy as np
        import matplotlib.pyplot as plt

        sia = SentimentIntensityAnalyzer()

        sentiment_scores = []
        event_counts = []
        sentence_indices = []

        crime_event_indices = [event['sentence_idx'] for event in self.events if event['event_type'] == 'crime']
        investigation_event_indices = [event['sentence_idx'] for event in self.events if event['event_type'] == 'investigation']
        neutral_event_indices = [event['sentence_idx'] for event in self.events if event['event_type'] == 'neutral']

        total_sentences = len(self.sentences)

        for idx, sentence in enumerate(self.sentences):
            # Sentiment score
            sentiment = sia.polarity_scores(sentence)['compound']
            sentiment_scores.append(sentiment)
            sentence_indices.append(idx)

            event_count = 0
            if idx in crime_event_indices:
                event_count += 1
            if idx in investigation_event_indices:
                event_count += 1
            if idx in neutral_event_indices:
                event_count += 1
            event_counts.append(event_count)

        window_size = max(1, int(total_sentences * 0.05))
        sentiment_scores_smoothed = np.convolve(sentiment_scores, np.ones(window_size)/window_size, mode='valid')
        event_counts_smoothed = np.convolve(event_counts, np.ones(window_size)/window_size, mode='valid')
        indices_smoothed = np.arange(len(sentiment_scores_smoothed))
        sentiment_scores_normalized = (sentiment_scores_smoothed - np.min(sentiment_scores_smoothed)) / (np.max(sentiment_scores_smoothed) - np.min(sentiment_scores_smoothed))
        event_counts_normalized = (event_counts_smoothed - np.min(event_counts_smoothed)) / (np.max(event_counts_smoothed) - np.min(event_counts_smoothed))
        combined_scores = (sentiment_scores_normalized * 0.5) + (event_counts_normalized * 0.5)
        peak_idx = np.argmax(combined_scores)
        peak_value = combined_scores[peak_idx]
        exposition_end = next((i for i, score in enumerate(combined_scores[:peak_idx]) if score > np.mean(combined_scores[:peak_idx])), int(0.1 * len(combined_scores)))
        falling_action_start = peak_idx + next((i for i, score in enumerate(combined_scores[peak_idx:]) if score < np.mean(combined_scores[peak_idx:])), int(0.1 * len(combined_scores)))
        resolution_start = int(0.9 * len(combined_scores))

        total_smoothed_sentences = len(sentiment_scores_smoothed)
        plot_structure = {
            'Exposition': (0, exposition_end),
            'Rising Action': (exposition_end, peak_idx),
            'Climax': (peak_idx, peak_idx + 1),
            'Falling Action': (peak_idx + 1, resolution_start),
            'Resolution': (resolution_start, total_smoothed_sentences - 1)
        }

        # Print plot structure summary
        #print("Plot Structure Identification:")
        #for phase, (start_idx, end_idx) in plot_structure.items():
        #   start_sentence = self.sentences[start_idx]
        #   end_sentence = self.sentences[end_idx]
        #   print(f"\n{phase} (Sentences {start_idx} to {end_idx}):")
        #   print(f"Start: {start_sentence[:75]}...")
        #   print(f"End: {end_sentence[:75]}...")

        # Plot the combined scores with plot structure demarcations
        #plt.figure(figsize=(12, 6))
        #plt.plot(indices_smoothed, combined_scores, label='Combined Sentiment and Event Score')
        #plt.axvline(x=exposition_end, color='green', linestyle='--', label='Exposition End')
        #plt.axvline(x=peak_idx, color='red', linestyle='--', label='Climax')
        #plt.axvline(x=falling_action_start, color='orange', linestyle='--', label='Falling Action Start')
        #plt.axvline(x=resolution_start, color='purple', linestyle='--', label='Resolution Start')
        #plt.xlabel('Smoothed Sentence Index')
        #plt.ylabel('Combined Score')
        #plt.title('Plot Structure Identification')
        #plt.legend()
        #plt.show()
        #self.plot_structure = plot_structure


    def pre_process(self):
        if self.__get_book() == 0:
            self.__normalize()
            self.__extract_names()
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
        self.__identify_plot_structure()


def get_event_map():
    return {
        # Crime Occurrence
        'murder': 'crime',
        'kill': 'crime',
        'poison': 'crime',
        'assassinate': 'crime',
        'slay': 'crime',
        'shoot': 'crime',
        'stab': 'crime',
        'strangle': 'crime',
        'steal': 'crime',
        'thief': 'crime',
        'rob': 'crime',
        'robbery': 'crime',
        'burgle': 'crime',
        'burglary': 'crime',
        'theft': 'crime',
        'loot': 'crime',
        'pilfer': 'crime',
        'shoplift': 'crime',
        'pickpocket': 'crime',
        'snatch': 'crime',
        'swindle': 'crime',
        'embezzle': 'crime',
        'defraud': 'crime',
        'plunder': 'crime',
        'smuggle': 'crime',
        'extort': 'crime',
        'blackmail': 'crime',
        'heist': 'crime',
        'kidnap': 'crime',
        'abduct': 'crime',
        'arson': 'crime',
        'vandalize': 'crime',
        'fraud': 'crime',
        'counterfeit': 'crime',
        'forge': 'crime',
        'dead': 'crime',
        'ambush': 'crime',
        'betray': 'crime',
        'deceive': 'crime',
        'lie': 'crime',
        'disguise': 'crime',
        'manipulate': 'crime',
        'sabotage': 'crime',
        'conspire': 'crime',
        'plot': 'crime',
        'scheme': 'crime',
        'threaten': 'crime',
        'attack': 'crime',
        'assault': 'crime',
        'brawl': 'crime',
        'fear': 'crime',
        'argue': 'crime',

        # Discovery and Investigation
        'discover': 'investigation',
        'find': 'investigation',
        'uncover': 'investigation',
        'investigate': 'investigation',
        'search': 'investigation',
        'probe': 'investigation',
        'examine': 'investigation',
        'inspect': 'investigation',
        'observe': 'investigation',
        'watch': 'investigation',
        'monitor': 'investigation',
        'spy': 'investigation',
        'eavesdrop': 'investigation',
        'surveil': 'investigation',
        'interrogate': 'investigation',
        'detect': 'investigation',
        'pursue': 'investigation',
        'track': 'investigation',
        'trace': 'investigation',
        'question': 'investigation',
        'report': 'investigation',
        'witness': 'investigation',
        'follow': 'investigation',

        # Suspicion and Accusation
        'suspect': 'investigation',
        'accuse': 'investigation',
        'blame': 'investigation',
        'denounce': 'investigation',
        'allege': 'investigation',
        'condemn': 'investigation',

        # Legal Actions
        'arrest': 'investigation',
        'charge': 'investigation',
        'trial': 'investigation',
        'testify': 'investigation',
        'convict': 'investigation',
        'sentence': 'investigation',
        'indict': 'investigation',
        'prosecute': 'investigation',
        'acquit': 'investigation',

        # Confession and Revelation
        'confess': 'crime',
        'admit': 'crime',
        'reveal': 'crime',
        'expose': 'investigation',
        'disclose': 'investigation',
        'divulge': 'investigation',

        # Confrontation and Conflict
        'confront': 'investigation',
        'fight': 'crime',
        'challenge': 'investigation',

        # Escape and Evasion
        'escape': 'crime',
        'flee': 'crime',
        'run': 'crime',
        'hide': 'crime',
        'evade': 'crime',
        'elude': 'crime',

        # Planning and Strategy
        'plan': 'investigation',
        'organize': 'investigation',
        'prepare': 'investigation',
        'devise': 'investigation',

        # Surveillance and Observation
        'inspect': 'investigation',
        'observe': 'investigation',
        'examine': 'investigation',
        'watch': 'investigation',
        'monitor': 'investigation',
        'spy': 'investigation',
        'eavesdrop': 'investigation',
        'surveil': 'investigation',

        # Emotional and Psychological Actions
        'warn': 'neutral',
        'fear': 'neutral',
        'worry': 'neutral',
        'panic': 'neutral',
        'regret': 'neutral',
        'admire': 'neutral',

        # Miscellaneous Relevant Actions
        'gather': 'neutral',
        'assemble': 'neutral',
        'proclaim': 'neutral',
        'announce': 'neutral',
    }
