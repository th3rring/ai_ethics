from pathlib import Path
from sys import argv

from ai_sentiment.nlp import AsentSentimentClassifier, SentimentClassifier
from ai_sentiment.scraper import CSVScraper

for data_path in argv[1:]:
  try:
    # Path to scrapes to look at
    csv_path = Path(data_path)

    # Project root dir
    project_dir = Path(__file__).parents[1]

    print(f"Loading data from {project_dir / csv_path}")
    csv_scraper = CSVScraper()
    csv_scraper.queueCSV(project_dir / csv_path)

    # Extract text articles
    targets = csv_scraper.scrapeAll()
    target_path = Path(f"{csv_path}_targets.yml")
    csv_scraper.dumpTargets(project_dir / target_path, targets)

    asent_results_file = f"{project_dir / 'new_data_results' / csv_path.stem}_asent.csv"
    asent_nlp = AsentSentimentClassifier()
    results = asent_nlp.processList(targets)
    print(f"Dumping asent results to {asent_results_file}")
    asent_nlp.dumpResults(asent_results_file, results)

    textblob_results_file = f"{project_dir / 'new_data_results' / csv_path.stem}_textblob.csv"
    textblob_nlp = SentimentClassifier()
    results = textblob_nlp.processList(targets)
    print(f"Dumping textblob results to {textblob_results_file}")
    textblob_nlp.dumpResults(textblob_results_file, results)
  except Exception as e:
    print(f"Error trying to process {csv_path}: {e}")
