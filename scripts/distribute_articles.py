from csv import DictReader, DictWriter
from dataclasses import dataclass
from json import dump
from pathlib import Path
from random import shuffle
from shutil import move
from subprocess import run
from tempfile import NamedTemporaryFile

from fire import Fire


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


def load_articles(csv_path: Path) -> list[Article]:
  with open(csv_path) as csv_file:
    rows = DictReader(csv_file)
    articles = [Article(0, "Anonymous", row["Title"], row["Body"]) for row in rows]
    shuffle(articles)
    for i, article in enumerate(articles):
      article.id = i

    return articles


def main(csv_path: Path, output_path: Path):
  generate_document(load_articles(csv_path), output_path)


if __name__ == "__main__":
  Fire(main)
