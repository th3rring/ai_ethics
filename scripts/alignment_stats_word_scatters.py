from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from re import compile
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
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


@dataclass(slots = True)
class Counts:
  articles: int = 0
  words: int = 0

  def __add__(self, other: Counts) -> Counts:
    return Counts(self.articles + other.articles, self.words + other.words)


def group_by_source(
    results: dict[str, DataFrame]
) -> tuple[DataFrame, dict[str, list[str]], dict[str, list[str]], dict[str, Counts]]:
  aggregates = []
  positive_words = defaultdict(list)
  negative_words = defaultdict(list)
  counts = defaultdict(Counts)
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
    counts[source_name] += Counts(
        len(data["titles"]), sum(len(re.findall(r'\w+', b)) for b in data["body_contents"])
    )

  return DataFrame(aggregates), positive_words, negative_words, counts


textblob_results, textblob_positive_words, textblob_negative_words, textblob_counts = group_by_source({
    path: SentimentClassifier.loadResults(path)
    for path in argv[1:]
    if "textblob" in path
})


def generate_scatterplots(words: dict[str, list[str]], source_path_format: str, aggregate_path_format: str):
  raise NotImplementedError("Need to expose per-word sentiment scores first!")
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


def plot_stats(counts: dict[str, Counts], alignment_path_format: str, aggregate_path: str):
  aggregate_counts = defaultdict(Counts)
  alignment_counts = defaultdict(list)
  for source_name, source_counts in counts.items():
    alignment = SOURCE_CLASSIFICATIONS[source_name]
    aggregate_counts[alignment] += source_counts
    alignment_counts[alignment].append((source_name, source_counts))

  label_locs = np.arange(len(aggregate_counts))
  width = 0.25
  fig, ax = plt.subplots(layout = 'constrained')
  alignments = []
  article_counts = []
  word_counts = []

  for alignment in Classification:
    a_counts = aggregate_counts[alignment]
    alignments.append(alignment.name.lower())
    article_counts.append(a_counts.articles)
    word_counts.append(a_counts.words / 1000)

  article_rects = ax.bar(label_locs, article_counts, width = width, align = 'edge', label = "Articles")
  word_rects = ax.bar(
      label_locs + width + 0.1, word_counts, width = width, align = 'edge', label = "1k Words"
  )
  ax.bar_label(article_rects, padding = 3)
  ax.bar_label(word_rects, padding = 3)

  ax.set_ylabel('Number of articles/thousands of words per alignment')
  ax.set_title('Article and word counts by political alignment')
  ax.set_xticks(label_locs + width, alignments)
  ax.legend(loc = 'upper left', ncols = 2)
  plt.savefig(aggregate_path)
  plt.close()

  for alignment, source_counts in alignment_counts.items():
    label_locs = np.arange(len(source_counts))
    width = 0.25
    fig, ax = plt.subplots(layout = 'constrained')
    sources = []
    article_counts = []
    word_counts = []

    for source, s_counts in source_counts:
      sources.append(source)
      article_counts.append(s_counts.articles)
      word_counts.append(s_counts.words / 1000)

    article_rects = ax.bar(label_locs, article_counts, width = width, align = 'edge', label = "Articles")
    word_rects = ax.bar(
        label_locs + width + 0.1, word_counts, width = width, align = 'edge', label = "1k Words"
    )
    ax.bar_label(article_rects, padding = 3)
    ax.bar_label(word_rects, padding = 3)

    ax.set_ylabel(f'Number of articles/thousands of words per {alignment.name.lower()} source')
    ax.set_title(f'Article and word counts by {alignment.name.lower()} source')
    ax.set_xticks(label_locs + width, sources)
    ax.legend(loc = 'upper left', ncols = 2)

    plt.savefig(alignment_path_format.format(alignment = alignment.name.lower()))
    plt.close()


for alignment in set(SOURCE_CLASSIFICATIONS.values()):
  Path(f"wordclouds/{alignment.name.lower()}").mkdir(parents = True, exist_ok = True)
  Path("stats").mkdir(parents = True, exist_ok = True)

# TODO: Word scatter plots
plot_stats(textblob_counts, "stats/{alignment}_stats.pdf", "stats/aggregate_stats.pdf")
