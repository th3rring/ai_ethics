from pathlib import Path
from sys import argv

import matplotlib.pyplot as plt
import seaborn as sns

from ai_sentiment.nlp import SentimentClassifier

results = {path: SentimentClassifier.loadResults(path) for path in argv[1:]}
