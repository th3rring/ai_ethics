from collections import defaultdict
from enum import Enum
from re import compile
from sys import argv

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from pandas import DataFrame
from wordcloud import WordCloud

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


def group_by_source(
    results: dict[str, DataFrame]
) -> tuple[DataFrame, dict[str, list[str]], dict[str, list[str]]]:
  aggregates = []
  positive_words = defaultdict(list)
  negative_words = defaultdict(list)
  for path, data in results.items():
    source_name = SOURCE_REGEX.search(path).group(1)  # type: ignore
    if source_name in NAME_MAP:
      source_name = NAME_MAP[source_name]

    aggregates.extend([{
        "source_name": source_name,
        "classification": SOURCE_CLASSIFICATIONS[source_name]._name_,
        "score": score
    } for score in data["sentiment_score"]])
    positive_words[source_name].extend(w for ws in data["positive_words"] for w in eval(ws))
    negative_words[source_name].extend(w for ws in data["negative_words"] for w in eval(ws))

  return DataFrame(aggregates), positive_words, negative_words


asent_results, asent_positive_words, asent_negative_words = group_by_source({
    path: SentimentClassifier.loadResults(path)
    for path in argv[1:]
    if "asent" in path
})
textblob_results, textblob_positive_words, textblob_negative_words = group_by_source({
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


def generate_wordcloud(words: dict[str, list[str]], format: str):
  for source_name, source_words in words.items():
    wordcloud = WordCloud(
        max_font_size = 100,
        max_words = 100,
        background_color = "white",
        scale = 2,
        width = 800,
        height = 400
    ).generate(' '.join(source_words))
    wordcloud.to_file(format.format(source_name = source_name))


asent_fig, asent_ax = plt.subplots()
plot_results(asent_ax, asent_results, "Sentiment Distribution by Source (asent)")
asent_fig.savefig("truncated_asent_results.pdf")
textblob_fig, textblob_ax = plt.subplots()
plot_results(textblob_ax, textblob_results, "Sentiment Distribution by Source (textblob)")
textblob_fig.savefig("truncated_textblob_results.pdf")

generate_wordcloud(asent_positive_words, "wordclouds/{source_name}_asent_positive_truncated.pdf")
generate_wordcloud(asent_negative_words, "wordclouds/{source_name}_asent_negative_truncated.pdf")
generate_wordcloud(textblob_positive_words, "wordclouds/{source_name}_textblob_positive_truncated.pdf")
generate_wordcloud(textblob_negative_words, "wordclouds/{source_name}_textblob_negative_truncated.pdf")
