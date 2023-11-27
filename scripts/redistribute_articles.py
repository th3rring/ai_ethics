from __future__ import annotations

from collections import defaultdict
from csv import DictReader
from dataclasses import dataclass
from itertools import groupby
from json import dump, load
from pathlib import Path

from fire import Fire
from rich import print


@dataclass(slots = True)
class Article:
  source: str
  title: str
  ids: tuple[int, ...]
  coders: tuple[int, ...]

  def __hash__(self) -> int:
    return hash((self.source, self.title))


def load_articles(article_map_path: Path) -> dict[int, Article]:
  with open(article_map_path) as article_map_file:
    article_map_data = load(article_map_file)

  article_groups = defaultdict(list)
  for article in article_map_data:
    article_groups[article["title"]].append(article)

  articles = {}
  for title, group in article_groups.items():
    article = Article(
        group[0]["source"], title, tuple(a["id"] for a in group), tuple(a["coder"] for a in group)
    )
    for a in group:
      articles[a["id"]] = article

  return articles


def already_coded(row) -> bool:
  return any((row["Valence"], row["Hopes"], row["Fears"], row["Frame"], row["Credibility"], row["Detail"]))


def load_coding_progress(coding_csv_path: Path) -> list[int]:
  with open(coding_csv_path) as coding_csv_file:
    reader = DictReader(coding_csv_file)
    return [int(r["Article #"]) for r in reader if already_coded(r)]


def main(coding_csv_path: Path, article_map_path: Path, total_num_articles: int):
  coding_csv_path = Path(coding_csv_path)
  article_map_path = Path(article_map_path)
  articles = load_articles(article_map_path)
  coded_article_ids = load_coding_progress(coding_csv_path)
  print(f"{len(coded_article_ids)} / {len(articles)} articles already coded")
  coded_articles = {articles[idx] for idx in coded_article_ids}
  print(f"{len(coded_articles)} unique articles already coded")
  source_articles = { s: list(a) for s, a in groupby(sorted(articles.values(), key=lambda a: a.source), key = lambda a: a.source) }
  source_counts = { s: len(a) for s, a in source_articles.items() }
  coded_source_counts = defaultdict(lambda: 0)
  for article in coded_articles:
    coded_source_counts[article.source] += 1

  for source in source_counts:
    print(f"Coded {coded_source_counts[source]} / {source_counts[source]} for {source}")

if __name__ == "__main__":
  Fire(main)
