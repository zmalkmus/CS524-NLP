# Normalization Strategy

## Text Hierarchy Split

The goal of splitting the text into this hierarchy is to make the text more manageable during processing 
as well as preserve enough context for making determinations about the story.

Chapter -> Paragraph -> Sentence

## Normalization

### Standard

* Remove all punctuation
* Remove all special characters
* Set everything to lower case

### Special Cases

* Hyphenated words should be treated as two words (e.g. "so-called" turns into "so called")
* Words with special formatting in text (e.g. italicized words represented as _word_) should have the formatting removed
* Special versions of an english character (e.g. a carroted 'a') should use the normal form of the letter
* A "--" at the end of a sentence or paragraph represents a character or thought getting interrupted and should signify the end of the sentence
* A "--" in the middle of a sentence represents a pause, and should be treated as a space within the sentence
* Special names of places or things with special characters within them should have the special characters removed (e.g. "Koh-i-noor" to "Kohinoor")
* Puncuation followed by a quotation mark (e.g. '?"') will not be assumed to be the end of a sentence unless the following character is a newline (\n) 

# Processing Techniques

## Name Recording

* Will look for names through text by comparing each word to a names database
* Names will be stored in a unique list to keep track of all characters
* If a name immediately follows another (e.g. John Doe), it will be considered a first and last name
* No duplicate names will be assumed

## Text Evaluation Strategy

Will be filled out once algorithm finalized
