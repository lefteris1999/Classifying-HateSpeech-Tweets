Eleftherios Konstantinou NLP coursework

important libraries used:

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import svm
import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from keras.models import load_model


DATASET Link: 1) https://huggingface.co/datasets/tweets_hate_speech_detection/viewer/default/train?p=319
	      2) https://huggingface.co/datasets/hate_speech_offensive

** Datasets should be able to be loaded by running the .ipynb code