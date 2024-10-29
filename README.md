# CS524 - NLP Project 1

#### Team: 7 

#### Author: G. K. Chesterton 

#### Task: Data Science Plot Analysis

#### DUE: OCTOBER 28

#### Installation

1. Requires a modern Python environment (3.12 is what was used but it should work with similar versions)
2. Run build.sh to install all python requirements in requirements.txt and install spacy en_core_web_sm 

If the build script does not work, install the pip packages manually with `pip install -r requirements.txt` and install spacy package with `python3 -m spacy download en_core_web_sm`

## Project Instructions

### Goals and the Task

The goal of this assignment is to design and implement a statistical model to analyze crime novel plots and predict key elements such as the primary perpetrator, the progression of the plot, and major turning points. You will work with a dataset of assigned author’s crime novels to explore narrative patterns and dependencies between different character roles, events, and textual features.

This assignment will involve data extraction, feature engineering, statistical modeling, and evaluation of results.

### Steps

#### Data Exploration

* Get novels in text format from Project Guttenberg and then prepare them for analysis (normalization and tokenization of your choice).
* Study the provided crime novels and their annotations.
* Examine how often key characters (protagonist, antagonist, victim, etc.) are introduced, the sequence of events, and common narrative structures.
* Analyze differences across different novels
* Prepare a short report (2 pages) summarizing key insights and narrative patterns.

#### Feature Engineering: Extract features from the text for plot analysis. Suggested features include:

* Character-based features
  * Time of first mention for each major character.
  * Co-occurrences between characters in the same scene.
  * Sentiment analysis of descriptions or dialogue related to characters.
* Event-based features
  * Timestamp or position of when the crime is first introduced.
  * Frequency and distribution of crime-related keywords.
  * Crime scene descriptions and their detailed analysis (e.g., adjectives used, tone).
* Structural features
  * Chapter/scene breakdown.
  * Transitions between major narrative points (e.g., calm scenes followed by action scenes).

#### Statistical Modeling:

* Plot Progression Model -- Develop a statistical or machine learning model — custom or based on known models (e.g., Bayesian Network) that predicts the likelihood of different plot developments as the novel progresses. The model should account for typical narrative arcs and the interplay of characters and events.
* Perpetrator Prediction Model -- Train a supervised model (e.g., logistic regression, random forest, or neural network) to predict the primary perpetrator using features such as:
  * Position and role of character first appearance.
  * How often they interact with the protagonist.
  * Sentiment/tone of language used around these characters.

#### Evaluation

* Perform cross-validation to assess the generalization of your models.
* Compare your predictions with the true outcomes (e.g., who the actual perpetrator is) and evaluate the performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score).
* Visualize the plot structures and the results of the predictions to highlight where the models succeed or fail.

#### Discussion and Analysis

* Analyze the output of your models. Where did they perform well? What kinds of errors were made?
* Discuss how narrative patterns differ between novels and whether it impacts the effectiveness of your models.
* Propose enhancements to your approach based on the results.