from dataclasses import dataclass
from typing import List


@dataclass
class ClassificationTarget:
    title: str
    body: str
    tags: List[str]

    @staticmethod
    def yamlRepresenter(dumper, data):
        """Static method passed to yaml to dump objects"""
        return dumper.represent_mapping('!ClassificationTarget',
                                        {'title': data.title,
                                         'body': data.body,
                                         'tags': data.tags})

    @staticmethod
    def yamlConstructor(loader, node):
        """Static method passed to yaml to load objects"""
        fields = loader.construct_mapping(node, deep=True)
        return ClassificationTarget(**fields)


@dataclass
class ClassificationResult:
    target: ClassificationTarget
    sentiment_score: float
    positive_words: List[str]
    negative_words: List[str]
