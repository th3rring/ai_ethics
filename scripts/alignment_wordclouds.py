from collections import defaultdict
from enum import Enum
from pathlib import Path
from re import compile
from sys import argv

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


textblob_results, textblob_positive_words, textblob_negative_words = group_by_source({
    path: SentimentClassifier.loadResults(path)
    for path in argv[1:]
    if "textblob" in path
})


def generate_wordclouds(words: dict[str, list[str]], source_path_format: str, aggregate_path_format: str):
  aggregate_words = defaultdict(set)
  for source_name, source_words in words.items():
    alignment = SOURCE_CLASSIFICATIONS[source_name]
    aggregate_words[alignment].update(source_words)
    wordcloud = WordCloud(
        max_font_size = 100,
        max_words = 100,
        background_color = "white",
        scale = 2,
        width = 1920,
        height = 1080
    ).generate(' '.join(source_words))
    wordcloud.to_file(
        source_path_format.format(alignment = alignment.name.lower(), source_name = source_name)
    )

  for alignment, alignment_words in aggregate_words.items():
    wordcloud = WordCloud(
        max_font_size = 100,
        max_words = 100,
        background_color = "white",
        scale = 2,
        width = 1920,
        height = 1080
    ).generate(' '.join(alignment_words))
    wordcloud.to_file(aggregate_path_format.format(alignment = alignment.name.lower()))

for alignment in set(SOURCE_CLASSIFICATIONS.values()):
  Path(f"wordclouds/{alignment.name.lower()}").mkdir(parents=True, exist_ok=True)

generate_wordclouds(
    textblob_positive_words,
    "wordclouds/{alignment}/{source_name}_textblob_positive.pdf",
    "wordclouds/{alignment}/aggregate_textblob_positive.pdf"
)
generate_wordclouds(
    textblob_negative_words,
    "wordclouds/{alignment}/{source_name}_textblob_negative.pdf",
    "wordclouds/{alignment}/aggregate_textblob_negative.pdf"
)
