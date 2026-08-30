from dataclasses import dataclass
from datetime import datetime


@dataclass
class RawTestimonial:
    source: str
    url: str
    title: str
    text: str
    author: str
    created_at: datetime