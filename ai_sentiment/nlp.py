import pickle
from pathlib import Path
from typing import List

import asent
import pandas as pd
import spacy
# from spacy import en_core_web_trf
from spacytextblob.spacytextblob import SpacyTextBlob

from ai_sentiment.data import ClassificationResult, ClassificationTarget


class SentimentClassifier:

  def __init__(self, pipeline = "en_core_web_trf"):
    """Init for NLP sentiment classifier

        Ensure that you've downloaded a model! Here, we default to the
        transformer model, https://spacy.io/models/en#en_core_web_trf"""

    self.nlp = spacy.load(pipeline)

    # self.nlp = en_core_web_trf.load()

    # TODO Check that this doesn't block the transformer model from running
    self.nlp.add_pipe('spacytextblob')

  def process(self, target: ClassificationTarget) -> ClassificationResult:
    """Call NLP pipeline on target

        Args:
            target: Item to classify, data class from this library

        Return:
            Result of classification!"""

    # Process!
    # Taken from https://importsem.com/evaluate-sentiment-analysis-in-bulk-with-spacy-and-python/
    doc = self.nlp(target.body)
    sentiment = doc._.blob.polarity
    sentiment = round(sentiment, 2)

    # Get classified words
    positive_words = []
    negative_words = []

    for x in doc._.blob.sentiment_assessments.assessments:
      if x[1] > 0:
        positive_words.append(x[0][0])
      elif x[1] < 0:
        negative_words.append(x[0][0])
      else:
        pass

    # Return classification
    return ClassificationResult(target, sentiment, positive_words, negative_words)

  def processList(self, targets: List[ClassificationTarget]) -> List[ClassificationResult]:
    """Calls NLP pipeline on a list of targets.

        Args:
            targets: A list of items to classify, each element is a
                ClassificationTarget data class

        Return:
            List of results from classification"""

    return [self.process(t) for t in targets]

  @staticmethod
  def dumpResults(filename: str, results: List[ClassificationResult], serialize = False):
    """Static method that dumps a list of results to a CSV file

        Args:
            filename: String filename for output csv, excludes file extension
            results: List of results from classification
            serialize: A boolean flag to enable outputting a
                serialized dataframe as well

        """

    filepath = Path(filename)

    # Create empty data frame
    df = pd.DataFrame()

    df["titles"] = [r.target.title for r in results]
    df["body_contents"] = [r.target.body for r in results]
    df["tags"] = [r.target.tags for r in results]
    df["sentiment_score"] = [r.sentiment_score for r in results]
    df["positive_words"] = [r.positive_words for r in results]
    df["negative_words"] = [r.negative_words for r in results]

    df.to_csv(filepath.with_suffix(".csv"))

    if serialize:
      with open(filepath.with_suffix(".pickle"), 'wb') as f:
        pickle.dump(df, f)

  @staticmethod
  def loadResults(filename: str):
    if type(filename) == str:
      filename = Path(filename)
    df = pd.read_csv(filename)
    return df

  @staticmethod
  def combineResults(filenames: List[str], write_filename: str):
    dfs = [SentimentClassifier.loadResults(f) for f in filenames]
    df = pd.concat(dfs)

    df.to_csv(Path(write_filename).with_suffix(".csv"))

  def processListToFile(self, filename: str, targets: List[ClassificationTarget]):
    """Calls NLP pipeline on a list of targets and dump results to a CSV file

        Args:
            filename: String filename for output csv, exclues file extension
            targets: A list of items to classify, each element is a ClassificationTarget data class
            """

    results = self.processList(targets)
    self.dumpResults(filename, results)


class AsentSentimentClassifier(SentimentClassifier):

  def __init__(self):
    self.nlp = spacy.load("en_core_web_trf")
    self.nlp.add_pipe('asent_en_v1')

  def process(self, target: ClassificationTarget) -> ClassificationResult:
    # TODO: Use Asent's visualizers
    # TODO: Document and sentence-level polarity
    doc = self.nlp(target.body)
    sentiment = doc._.polarity.compound
    sentiment = round(sentiment, 2)

    # Get classified words
    positive_words = []
    negative_words = []

    for x in doc:
      x_pol = x._.polarity
      if x_pol.polarity > 0:
        positive_words.append(x.text)
      elif x_pol.polarity < 0:
        negative_words.append(x.text)

    # Return classification
    return ClassificationResult(target, sentiment, positive_words, negative_words)
