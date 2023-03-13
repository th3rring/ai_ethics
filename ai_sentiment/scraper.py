from typing import List
from pathlib import Path
from abc import ABC, abstractmethod
from collections import deque, namedtuple
import json
import re
import os
from dotenv import load_dotenv

from ai_sentiment.data import ClassificationTarget 

# Imports for reddit scraping
import praw
from URS.urs.praw_scrapers.utils.Objectify import Objectify
from URS.urs.praw_scrapers.static_scrapers.Comments import SortComments

# Imports for website scraper
import pandas as pd
from bs4 import BeautifulSoup
import requests


class Scraper(ABC):

    @abstractmethod
    def scrapeNext(self) -> ClassificationTarget:
        """Abstract base class that pops from internal queue and creates a classification target"""
        raise NotImplementedError

    def scrapeAll(self) -> List[ClassificationTarget]:
        """Scrape all entries in queue and return as list"""

        r = []
        while self.queue:
            r.append(self.scrapeNext())

        return r

    queue = deque()

RedditQueueElement = namedtuple("RedditQueueElement", ["title", "url", "tags"])

class RedditScraper(Scraper):

    def __init__(self, keywords: List[str]):
        """Initialize reddit scraper

        Must match at least one keyword to be added to processing queue

        Args:
            keywords: List of regex patterns to match against post titles
        """
        # List of keywords to be 
        self.keywords = keywords

        load_dotenv()

        self.reddit_ = praw.Reddit(
            client_id = os.getenv("CLIENT_ID"),
            client_secret = os.getenv("CLIENT_SECRET"),
            user_agent = os.getenv("USER_AGENT"),
            username = os.getenv("REDDIT_USERNAME"),
            password = os.getenv("REDDIT_PASSWORD")
        )

        
    def queuePostsJson(self, path: Path):
        """Process URS post JSON files and pull comments from posts that meet keywords

        Args:
            path: a Pathlib path to the URS json file to be parsed

        Uses URS under the hood on posts that meet keyword filter, due to limitations of that tool this method will not queue
        """

        if not path.exists():
            raise OSError(f"URS results at {path} not found")

        # Attempt to load data at path
        data = None
        with open(path, "r") as read_file:
            data = json.loads(read_file.read())

        # Look through all posts
        for post in data["data"]:

            # If this post has no comments, move on
            if post["num_comments"] == 0:
                continue

            # Look through all keywords
            for kw in self.keywords:
                # attempt to match
                match = re.search(kw, post['title'])

                # If we have a match, add to queue and go to next post
                if match:
                    # Add current post to processing queue, add subreddit and post flair as tag
                    self.queue.append(RedditQueueElement(post["title"], "https://www.reddit.com" + post["permalink"], ["reddit_post", "r/" + data["scrape_settings"]["subreddit"], post["link_flair_text"]]))
                    break

    def scrapeNext(self) -> ClassificationTarget:
        """Create a classification target for the next request in the queue"""

        # Get next element in queue
        cur_element = self.queue.popleft()

        # Create praw submission
        submission = self.reddit_.submission(url = cur_element.url)

        # Append all comments in order 
        comments = []
        for comment in submission.comments.list():
            # Some comments from praw have no author attribute for some reason
            try:
                comments.append(Objectify().make_comment(comment, False))
            except AttributeError:
                print("Caught invalid comment! skipping...")
                continue

        body_content = ""
        for comment_content in comments:
            body_content += comment_content["author"] + ": " + comment_content["body"] + ". "



        return ClassificationTarget(cur_element.title, body_content, cur_element.tags)


WebQueueElement = namedtuple("WebQueueElement", ["title", "url", "tags"])

class WebScraper(Scraper):

    def __init__(self):
        """Init for web scraper"""

    def queueWebsiteCSV(self, path: Path):
        """Process a CSV of website URLs and add them to the queue

        Args:
            path: A PathLib path to a csv file with three columns \"Title\", \"Addresses\", and \"Tags\" """

        df = pd.read_csv(path)
        titles = df["Titles"].tolist()
        urls = df["Addresses"].tolist()
        tags = df["Tags"].tolist()

        for title, url, tag in zip(titles, urls, tags):
            self.queue.append(WebQueueElement(title, url, tag))

    def scrapeNext(self) -> ClassificationTarget:
        """Create a classification target for the next request in the queue"""

        # Get next element in queue
        cur_element = self.queue.popleft()

        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36'}
        res = requests.get(cur_element.url, headers=headers)
        html_page = res.text

        soup = BeautifulSoup(html_page, 'html.parser')
        for script in soup(["script", "style","meta","label","header","footer"]):
          script.decompose()
        page_text = (soup.get_text()).lower()
        page_text = page_text.strip().replace("  ","")
        page_text = "".join([s for s in page_text.splitlines(True) if s.strip("\r\n")])

        return ClassificationTarget(cur_element.title, page_text, cur_element.tags)
