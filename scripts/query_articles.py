from __future__ import annotations

from collections import defaultdict
from csv import DictReader
from dataclasses import dataclass
from enum import Enum
from json import load
from pathlib import Path
from re import compile
from shutil import move
from subprocess import DEVNULL, check_call
from tempfile import NamedTemporaryFile

from anyascii import anyascii
from fire import Fire


class Classification(Enum):
  LEFT = "Left"
  LEANS_LEFT = "Leans Left"
  CENTER = "Center"
  LEANS_RIGHT = "Leans Right"
  RIGHT = "Right"


SOURCE_CLASSIFICATIONS = {
    "Daily Caller": Classification.RIGHT,
    "Daily Wire": Classification.RIGHT,
    "NY Post": Classification.RIGHT,
    "New York Post": Classification.RIGHT,
    "Epoch Times": Classification.LEANS_RIGHT,
    "Fox News Online": Classification.LEANS_RIGHT,
    "WSJ Opinion": Classification.LEANS_RIGHT,
    "Wall Street Journal Opinion": Classification.LEANS_RIGHT,
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

SOURCE_REGEX = compile(r"- ([^_]*).csv")


@dataclass(slots = True)
class Article:
  source: str
  title: str
  body: str
  ids: list[int]

  def __hash__(self) -> int:
    return hash(self.body)

  def __repr__(self) -> str:
    return f"Article(id={self.ids}, source={self.source}, title={self.title})"


def latex_escape(text: str) -> str:
  # Strip unicode linebreaks
  # text = "\n\n".join(text.splitlines())
  # There's a more efficient way to do this, but I'm lazy
  problem_characters = (("%", "\\%"), ("$", "\\$"), ("&", "\\&"), ("#", "\\#"), ("@", "\\@"), ("_", "\\_"))
  for c, r in problem_characters:
    text = text.replace(c, r)

  return text


def generate_document(query_field: str, query_term: str, articles: list[Article], output_path: Path):
  for article in articles:
    if article.title == "" or article.body == "":
      raise RuntimeError(f"Empty article: {article}")

  bodies = [
      f"\\section*{{``{latex_escape(article.title)}''---{latex_escape(article.source)}---{SOURCE_CLASSIFICATIONS[article.source].value}}}\n{latex_escape(article.body)}"
      for article in articles
  ]
  pages_string = "\n\\newpage\n".join(bodies)
  document = f"""\\documentclass[a4paper,10pt]{{article}}
\\usepackage[top=1in, bottom=1in, left=1in, right=1in, footskip = 1.0cm]{{geometry}}
\\title{{Articles with ``{query_term}'' in {query_field}}}
\\author{{}}
\\begin{{document}}
\\maketitle
{pages_string}
\\end{{document}}
"""

  with NamedTemporaryFile(suffix = ".tex", delete = False) as tf:
    tf.write(document.encode("ascii", "ignore"))
    tf.seek(0)
    check_call(["latexmk", "-pdf", "-interaction=nonstopmode", tf.name], cwd = "/tmp", stdout = DEVNULL)
    check_call(["latexmk", "-c"], cwd = "/tmp", stdout = DEVNULL)
    move(Path(tf.name).with_suffix(".pdf"), output_path)


def gather_ids(row, article_map) -> list[int]:
  articles = article_map[row["Title"]]
  return [a["id"] for a in articles]


def load_source(source: str, csv_path: Path, article_map) -> list[Article]:
  with open(csv_path) as csv_file:
    rows = DictReader(csv_file)
    return [
        Article(
            anyascii(source), anyascii(row["Title"]), anyascii(row["Body"]), gather_ids(row, article_map)
        ) for row in rows
    ]


def load_articles(source_csv_paths: list[Path], article_map_path: Path) -> list[Article]:
  with open(article_map_path) as article_map_file:
    article_map_data = load(article_map_file)

  article_map = defaultdict(list)
  for article in article_map_data:
    article_map[article["title"]].append(article)

  articles = []
  for path in source_csv_paths:
    source_name = path.stem
    if source_name in NAME_MAP:
      source_name = NAME_MAP[source_name]

    articles.extend(load_source(source_name, path, article_map))

  return articles


def has_term(
    article: Article, query_field: str, query_term, results: dict[int, dict[str, str | int | bool]]
) -> bool:
  for id_ in article.ids:
    if id_ not in results:
      continue

    field_value = results[id_][query_field]
    if query_term == field_value or query_term in field_value:
      return True

  return False


def remap_fields(r) -> dict[str, int | str | bool]:
  return {
      "id":
          r["What is the article's ID?"],
      "valence":
          r["What's the overall valence/sentiment of the article toward LLMs?"],
      "opinion":
          r["Does the author present their own opinions/the opinions of others or only report facts?"] ==
          "Opinions",
      "sources":
          r["What sources does the article cite?"],
      "hopes":
          r["What hopes (i.e., real or hypothesized positive impacts) does the article express about LLMs, if any?"
            ],
      "fears":
          r["What fears (i.e., real or hypothesized negative impacts) does the article express about LLMs, if any?"
            ],
      "frame":
          r["What is the article discussing? Which of these frames/questions does it answer? Only select more than one if absolutely necessary"
            ]
  }


def load_results(results_csv: Path) -> dict[int, dict[str, int | str]]:
  with open(results_csv) as results_file:
    rows = DictReader(results_file)
    return {
        int(row["What is the article's ID?"]): remap_fields(row)
        for row in rows
        if row["What is the article's ID?"] != ""
    }


def main(
    article_map: Path,
    results_csv: Path,
    article_csv_dir: Path,
    output_prefix: Path,
    query_field: str,
    query_term: str
):
  output_prefix = Path(output_prefix)
  output_prefix.mkdir(parents = True, exist_ok = True)
  article_csv_dir = Path(article_csv_dir)
  csv_paths = list(article_csv_dir.glob("*.csv"))
  articles = load_articles(csv_paths, Path(article_map))
  results_map = load_results(results_csv)
  query_articles = [
      article for article in articles if has_term(article, query_field, query_term, results_map)
  ]

  generate_document(
      query_field, query_term, query_articles, output_prefix / f"{query_field}_{query_term}_articles.pdf"
  )


if __name__ == "__main__":
  Fire(main)
