from collections import defaultdict
from csv import DictReader, DictWriter
from pathlib import Path
from sys import argv

sources = defaultdict(list)
normalized_names = {
    "WSJ, Opinion": "WSJ Opinion",
    "Washington post": "Washington Post",
    "NYT, Opinion, Letters": "New York Times Opinion Letters",
    "NYT": "New York Times",
    "NYT, Opinion": "New York Times Opinion",
    "WSJ": "Wall Street Journal",
    "NYPost": "NY Post",
    "Washington post, Opinion": "Washington Post Opinion",
    "Washington post, Opinion, Letters": "Washington Post Opinion Letters",
    "USAToday": "USA Today"
}
for data_path in argv[1:]:
  with open(Path(data_path)) as data_file:
    csv_data = DictReader(data_file)
    for row in csv_data:
      source = normalized_names.get(row["Tags"], row["Tags"])
      sources[source].append({
          "Title": row["Titles"],
          "Link": row["Addresses"],
          "Body": row["Body"],
          "Date": "",
          "Notes": "",
          "Author": "MISSING",
      })

field_names = ["Title", "Link", "Body", "Date", "Notes", "Author"]
for source, articles in sources.items():
  with open(f"new_data/Old Articles - {source}.csv", "w") as source_file:
    writer = DictWriter(source_file, fieldnames = field_names)
    writer.writeheader()
    writer.writerows(articles)
