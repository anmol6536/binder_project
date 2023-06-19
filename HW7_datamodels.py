import networkx as nx
import matplotlib.pyplot as plt

data = open('./datasets/fradulent_emails_clean.txt', 'r+').read()
data = data.split('From r ')

import re
from typing import Optional
import string
from datetime import datetime


def remove_punctuation(s: str) -> str:
    """Remove punctuation characters from a string."""
    return s.translate(str.maketrans('', '', string.punctuation))


class Mail:
    subject: Optional[str] = None
    return_email: Optional[str] = None
    to_email: Optional[str] = None
    date: Optional[datetime] = None
    body: Optional[str] = None

    def __init__(self, value: str) -> None:
        """Initialize a Mail object with the given value."""
        self.__value__ = value

    def extract_metadata(self) -> None:
        """Extract metadata (subject and return email) from the mail content."""
        subjects = re.findall(r'Subject: (.*)', self.__value__)
        if subjects:
            self.subject = remove_punctuation(''.join(subjects))

        return_email = re.search(r'Return-Path: .*', self.__value__)
        if return_email:
            self.return_email = self.__extract_email(return_email.group(0))

        date = re.search(r'Date: .*', self.__value__)
        if date:
            self.date = date.group(0).split(" ", 2)[2]

        to_email = re.search(r'To: .*', self.__value__)
        if to_email:
            self.to_email = self.__extract_email(to_email.group(0))

        self.body = self.__extract_body(self.__value__)
        self.body = self.__remove_html_tags(self.body)

        self.__value__ = re.sub(r'(\n){1,9}', '\n', self.__value__)

    @staticmethod
    def __extract_email(string: str) -> str:
        """Extract an email from a string."""
        email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', string)
        if email:
            return email.group(0)

    @staticmethod
    def __extract_body(string: str) -> str:
        """Extract the body of the mail from a string."""
        lines = string.split('\n')
        lines_to_search = lines[:200]
        body = []
        for line in lines_to_search:
            if re.search(r'^(\w+-?){1,4}:', line):
                continue
            body.append(line)
        body = [i for i in [*body, *lines[20:]] if i]
        return '\n'.join(body)

    @staticmethod
    def __remove_html_tags(string: str) -> str:
        """Remove HTML tags from a string."""
        return re.sub(r'<.*?>', '', string)

    @classmethod
    def from_string(cls, value: str) -> 'Mail':
        """Create a Mail object from a string."""
        return cls(value)

    def __repr__(self) -> str:
        """Return a string representation of the Mail object."""
        return f'Mail(subject={self.subject}, return_email={self.return_email},' \
               f' to_email={self.to_email}, date={self.date}, body={self.body})'

from tqdm import tqdm
mails = [Mail.from_string(m) for m in data if m]
_ = [i.extract_metadata() for i in tqdm(mails, total=len(mails))]
print("Number of mails", len(mails))
print(len(data))

edges = [(i.return_email, i.to_email) for i in mails if bool(i.return_email and i.to_email) & (i.return_email != i.to_email)]
nodes = list(set([i for j in edges for i in j]))

f, ax = plt.subplots(figsize=(20, 20))
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
d = nx.degree(G)
pos = nx.spring_layout(G, k=0.5, iterations=50)

scammer_email = [i.return_email for i in mails if bool(i.return_email) & (i.return_email in G.nodes)]
scam_victim_email = [i.to_email for i in mails if bool(i.to_email) & (i.to_email in G.nodes)]

nx.draw_networkx_nodes(G,
                       pos,
                       nodelist=scammer_email,
                       node_size=[d[v] * 100 for v in scammer_email],
                       node_color='blue',
                       edgecolors='black',
                       ax=ax)
nx.draw_networkx_nodes(G,
                       pos,
                       nodelist=scam_victim_email,
                       node_size=[d[v] * 100 for v in scam_victim_email],
                       node_color='red',
                       edgecolors='black',
                       ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

plt.savefig('./output_images/HW7_datamodels.png')

for email in sorted(dict(d), key=lambda x: d[x], reverse=True)[:10]:
    if email in scammer_email:
        print(f"({d[email]}){email.ljust(40, '-')}: Scammer")
    elif email in scam_victim_email:
        print(f"({d[email]}){email.ljust(40, '-')}: Scam victim")
    else:
        print(f"({d[email]}){email.ljust(40, '-')}: Neither")