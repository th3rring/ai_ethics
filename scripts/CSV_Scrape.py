from pathlib import Path
from sys import argv

import numpy as np

from ai_sentiment.nlp import SentimentClassifier
from ai_sentiment.scraper import CSVScraper

# Path to scrapes to look at
csv_path = Path(argv[1])

# Project root dir
project_dir = Path(__file__).parent

print(f"Loading data from {project_dir / csv_path}")
csv_scraper = CSVScraper()
csv_scraper.queueCSV(project_dir / csv_path)

# Extract text articles
targets = csv_scraper.scrapeAll()
target_path = Path(f"{csv_path}_targets.yml")
csv_scraper.dumpTargets(project_dir / target_path, targets)
dataset_name = csv_path.stem
split_N = int(argv[2])
split = np.array_split(np.asarray(targets), split_N)
for t in range(split_N):
  nlper = SentimentClassifier()
  results = nlper.processList(split[t])
  nlper.dumpResults(f"{dataset_name}_{t}_results", results)

combined_fn = f"{dataset_name}_results"
SentimentClassifier.combineResults([f"{dataset_name}_{t}_results.csv" for t in range(split_N)], combined_fn)
nlper = SentimentClassifier()
results = nlper.processList(targets)
nlper.dumpResults(combined_fn, results)
