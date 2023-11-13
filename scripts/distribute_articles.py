from __future__ import annotations

import re
from re import compile
import numpy as np
from csv import DictReader, DictWriter
from dataclasses import dataclass
from json import dump
from enum import Enum
from pathlib import Path
from random import shuffle
from shutil import move
from subprocess import run
from tempfile import NamedTemporaryFile

from fire import Fire

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
class Article:
  id: int
  source: str
  title: str
  body: str


def latex_escape(text: str) -> str:
  return text.replace("%", "\\%").replace("$", "\\$").replace("&", "\\&")


def generate_document(articles: list[Article], output_path: Path):
  bodies = [
      f"\\section{{``{latex_escape(article.title)}''}}\n{latex_escape(article.body)}"
      for article in articles
  ]
  pages_string = "\n\\newpage\n".join(bodies)
  document = f"""\\documentclass[a4paper,10pt]{{article}}
\\usepackage[top=1in, bottom=1in, left=1in, right=1in, footskip = 1.0cm]{{geometry}}
\\title{{Manual Coding Articles}}
\\author{{}}
\\begin{{document}}
\\maketitle
{pages_string}
\\end{{document}}
"""

  with NamedTemporaryFile(suffix = ".tex", delete=False) as tf:
    tf.write(document.encode())
    tf.seek(0)
    run(["latexmk", "-pdf", "-xelatex", "-interaction=nonstopmode", tf.name], cwd = "/tmp")
    run(["latexmk", "-c"], cwd = "/tmp")
    move(Path(tf.name).with_suffix(".pdf"), output_path)


def load_articles(source: str, csv_path: Path) -> list[Article]:
  with open(csv_path) as csv_file:
    rows = DictReader(csv_file)
    return [Article(0, source, row["Title"], row["Body"]) for row in rows]

def load_sources(source_csv_paths: list[Path]) -> dict[str, list[Article]]:
  source_articles = {}
  for path in source_csv_paths:
    source_name = SOURCE_REGEX.search(path).group(1)  # type: ignore
    if source_name in NAME_MAP:
      source_name = NAME_MAP[source_name]
      assert source_name not in source_articles, f"{source_name} already processed before {path}!"
      source_articles[source_name] = load_articles(source_name, path)
  
  return source_articles


def main(num_coders: int, output_prefix: Path, *csv_paths, num_coders_per_article: int = 2,):
  articles = load_sources(csv_paths)
  article_count = sum(len(source_articles) for source_articles in articles.values())


if __name__ == "__main__":
  Fire(main)
