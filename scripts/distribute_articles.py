from __future__ import annotations

from collections import defaultdict
from csv import DictReader, DictWriter
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain, cycle
from json import dump
from pathlib import Path
from random import shuffle
from re import compile
from shutil import move
from subprocess import DEVNULL, check_call
from tempfile import NamedTemporaryFile

from anyascii import anyascii
from fire import Fire
from more_itertools import batched, random_permutation, windowed


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

SOURCE_REGEX = compile(r"- ([^_]*).csv")


@dataclass(slots = True)
class Article:
  source: str
  title: str
  body: str
  ids: list[int] = field(default_factory = list)
  coders: list[int] = field(default_factory = list)

  def __hash__(self) -> int:
    return hash(self.body)

  def __repr__(self) -> str:
    return f"Article(id={self.ids}, source={self.source}, title={self.title}, coders={self.coders})"


def latex_escape(text: str) -> str:
  # Strip unicode linebreaks
  # text = "\n\n".join(text.splitlines())
  # There's a more efficient way to do this, but I'm lazy
  problem_characters = (("%", "\\%"), ("$", "\\$"), ("&", "\\&"), ("#", "\\#"), ("@", "\\@"), ("_", "\\_"))
  for c, r in problem_characters:
    text = text.replace(c, r)

  return text


def generate_document(coder_id: int, articles: list[dict[str, str]], output_path: Path):
  for article in articles:
    if article["title"] == "" or article["body"] == "":
      raise RuntimeError(f"Empty article: {article}")

  first_idx = int(articles[0]["id"])
  bodies = [
      f"\\section{{``{latex_escape(article['title'])}''}}\n{latex_escape(article['body'])}"
      for article in articles
  ]
  pages_string = "\n\\newpage\n".join(bodies)
  document = f"""\\documentclass[a4paper,10pt]{{article}}
\\usepackage[top=1in, bottom=1in, left=1in, right=1in, footskip = 1.0cm]{{geometry}}
\\title{{Manual Coding Articles - Coder {coder_id}}}
\\author{{}}
\\setcounter{{section}}{{{first_idx - 1}}}
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


def load_articles(source: str, csv_path: Path) -> list[Article]:
  with open(csv_path) as csv_file:
    rows = DictReader(csv_file)
    return [Article(anyascii(source), anyascii(row["Title"]), anyascii(row["Body"])) for row in rows]


def load_sources(source_csv_paths: list[Path]) -> dict[str, list[Article]]:
  source_articles = {}
  for path in source_csv_paths:
    source_name = path.stem
    if source_name in NAME_MAP:
      source_name = NAME_MAP[source_name]
      assert source_name not in source_articles, f"{source_name} already processed before {path}!"

    source_articles[source_name] = load_articles(source_name, path)

  return source_articles


def main(
    num_coders: int,
    output_prefix: Path,
    csv_dir: Path,
    num_coders_per_article: int = 2,
    titles_list_file: Path | None = None
):
  output_prefix = Path(output_prefix)
  output_prefix.mkdir(parents = True, exist_ok = True)
  csv_dir = Path(csv_dir)
  csv_paths = list(csv_dir.glob("*.csv"))
  if titles_list_file:
    titles_list_file = Path(titles_list_file)
    with open(titles_list_file) as title_data:
      titles = {anyascii(l).strip() for l in title_data.readlines()}
    articles = [a for a in chain.from_iterable(load_sources(csv_paths).values()) if a.title in titles]
    print(len(articles))
  else:
    articles = list(chain.from_iterable(load_sources(csv_paths).values()))

  shuffle(articles)
  # Handling indivisibility:
  remainder = len(articles) % num_coders
  if remainder == 0:
    divisible_articles = articles
  else:
    divisible_articles = articles[:-remainder]

  batches = [list(b) for b in batched(divisible_articles, len(articles) // num_coders)]
  residue = articles[-remainder:]
  for article, idx in zip(residue, random_permutation(range(num_coders)), strict = False):
    batches[idx].append(article)

  article_batches = windowed(cycle(batches), num_coders_per_article)
  article_idx = 1
  for i in range(1, num_coders + 1):
    coder_batch = next(article_batches)
    flattened_batch = list(set(chain.from_iterable(coder_batch)))  # type: ignore
    for article in flattened_batch:
      article.ids.append(article_idx)
      article_idx += 1
      article.coders.append(i)

  flattened_articles = []
  coder_articles = defaultdict(list)
  print(f"{len(articles)} articles")
  for article in articles:
    if len(article.coders) != num_coders_per_article:
      print(f"Warning: expected {num_coders_per_article} coders but got {len(article.coders)} for {article}")

    for ID, coder in zip(article.ids, article.coders, strict = True):
      article_metadata = { "id": ID, "coder": coder, "title": article.title, "source": article.source}
      flattened_articles.append(article_metadata)
      flattened_article = {
          "id": ID, "coder": coder, "title": article.title, "body": article.body, "source": article.source
      }
      coder_articles[coder].append(flattened_article)

  print(f"{len(flattened_articles)} coder articles")
  flattened_articles.sort(key = lambda a: a["id"])

  for coder, coder_batch in coder_articles.items():
    coder_batch.sort(key = lambda a: a["id"])
    print(f"Generating PDF for coder {coder}...")
    generate_document(coder, coder_batch, output_prefix / f"coder_{coder}.pdf")

  with open(output_prefix / "article_map.json", "w") as article_map_file:
    dump(flattened_articles, article_map_file)

  with open(output_prefix / "coding_data.csv", "w") as coding_data_file:
    coding_data_writer = DictWriter(
        coding_data_file, fieldnames = ["Valence", "Hopes", "Fears", "Frame", "Credibility", "Detail"]
    )
    coding_data_writer.writeheader()
    for article in flattened_articles:
      coding_data_writer.writerow({
          "Valence": "", "Hopes": "", "Fears": "", "Frame": "", "Credibility": "", "Detail": ""
      })


if __name__ == "__main__":
  Fire(main)
