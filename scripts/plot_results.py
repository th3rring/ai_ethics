from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from re import compile
from sys import argv

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from numpy import nan
from pandas import DataFrame

from ai_sentiment.nlp import SentimentClassifier


class Classification(Enum):
  LEFT = "#2e65a1"
  LEANS_LEFT = "#9dc8eb"
  CENTER = "#9766a0"
  LEANS_RIGHT = "#cb9a98"
  RIGHT = "#cb2127"


SOURCE_CLASSIFICATIONS = {
    "Daily Caller": Classification.RIGHT,
    "Daily Wire": Classification.RIGHT,
    "NY Post": Classification.RIGHT,
    "Epoch Times": Classification.LEANS_RIGHT,
    "Fox News Online": Classification.LEANS_RIGHT,
    "WSJ Opinion": Classification.LEANS_RIGHT,
    "Washington Examiner": Classification.LEANS_RIGHT,
    "Reuters": Classification.CENTER,
    "WSJ": Classification.CENTER,
    "Wall Street Journal": Classification.CENTER,
    "New York Times": Classification.LEANS_LEFT,
    "USA Today": Classification.LEANS_LEFT,
    "Washington Post": Classification.LEANS_LEFT,
    "Washington Post Opinion": Classification.LEANS_LEFT,
    "Washington Post Opinion Letters": Classification.LEANS_LEFT,
    "New York Times Opinion": Classification.LEFT,
    "New York Times Opinion Letters": Classification.LEFT,
    "New Yorker": Classification.LEFT,
    "The Atlantic": Classification.LEFT,
    "Vox": Classification.LEFT,
}

NAME_MAP = {
    "WSJ": "Wall Street Journal",
    "New York Times Opinion Letters": "New York Times Opinion",
    "Washington Post Opinion Letters": "Washington Post Opinion"
}

SOURCE_REGEX = compile(r"- ([^_]*)_")


def group_by_source(results: dict[str, DataFrame]) -> DataFrame:
  aggregates = []
  for path, data in results.items():
    source_name = SOURCE_REGEX.search(path).group(1)  # type: ignore
    if source_name in NAME_MAP:
      source_name = NAME_MAP[source_name]

    aggregates.extend([{
        "source_name": source_name,
        "classification": SOURCE_CLASSIFICATIONS[source_name]._name_,
        "score": score
    } for score in data["sentiment_score"]])

  return DataFrame(aggregates)


asent_results = group_by_source({
    path: SentimentClassifier.loadResults(path)
    for path in argv[1:]
    if "asent" in path
})
textblob_results = group_by_source({
    path: SentimentClassifier.loadResults(path)
    for path in argv[1:]
    if "textblob" in path
})


def plot_results(ax: Axes, results: DataFrame, title: str):
  ax.set_title(title)
  sns.boxplot(
      data = results,
      x = "score",
      y = "source_name",
      hue = "classification",
      palette = {c._name_: c._value_
                 for c in Classification},
      order = list(SOURCE_CLASSIFICATIONS.keys()),
      ax = ax
  )


asent_fig, asent_ax = plt.subplots()
plot_results(asent_ax, asent_results, "Sentiment Distribution by Source (asent)")
asent_fig.savefig("asent_results.pdf")
textblob_fig, textblob_ax = plt.subplots()
plot_results(textblob_ax, textblob_results, "Sentiment Distribution by Source (textblob)")
textblob_fig.savefig("textblob_results.pdf")

plt.show()
