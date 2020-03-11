import argparse
import re
import requests
import sys
import urllib

import bs4
import pandas


unclassified_cards = {}


class Rule(object):
    def __init__(self, name, required_cards):
        self.name = name
        self.required_cards = required_cards

    def test(self, name, cards):
        for required_card in self.required_cards:
            if required_card not in cards:
                return False
        return True


def default_name(name, url):
    return name


def name_from_cards(rule_file, default_name='Other'):
    with open(rule_file) as f:
        lines = f.readlines()
    rules = []
    for line in lines:
        lst = line.strip().split(',')
        rules.append(Rule(lst[0], lst[1:]))
    def apply_rules(name, url):
        request = requests.get(url)
        soup = bs4.BeautifulSoup(request.content, "html.parser")
        d_table = soup.find("table", attrs={"class": "deck-view-deck-table"})
        if d_table is None:
            print(f"Couldn't find/parse deck at url {url}")
            return name
        cards = []
        for row in d_table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) > 1:
                cards.append(cells[1].text.strip())
            elif 'sideboard' in cells[0].text.strip().lower():
                break
        for rule in rules:
            if rule.test(name, cards):
                return rule.name
        for card in cards:
            unclassified_cards[card] = unclassified_cards.get(card, 0) + 1
        return default_name
    return apply_rules


def mtggoldfish(url, deck_fn=default_name):
    request = requests.get(url)
    soup = bs4.BeautifulSoup(request.content, "html.parser")
    t_table = soup.find("table", attrs={"class": "table-tournament"})
    if t_table is None:
        print(f"Couldn't find tournament at url {url}")
        return None
    t_data = {'source': url}
    heading = soup.find("div", attrs={"class": "col-md-12"})
    if heading is not None:
        date = re.sub('^.*Date: ([-0-9]+).*$', '\\1', heading.text, flags=re.S)
        if re.match('^[-0-9]+$', date):
            t_data['date'] = date
        name = heading.find("h1")
        if name is not None:
            t_data['t_name'] = name.text.strip()
    entries = []
    for row in t_table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 4:
            entry = {
                'w': int(cells[0].text.strip()),
                'l': int(cells[1].text.strip()),
                'deck': cells[2].text.strip(),
                'player': cells[3].text.strip()
            }
            link_tag = cells[2].find("a")
            link = link_tag.get('href', None) if link_tag is not None else None
            if deck_fn is not None:
                entry['deck'] = deck_fn(entry['deck'], urllib.parse.urljoin(url, link))
            for key, val in t_data.items():
                entry[key] = val
            entries.append(entry)
    data = pandas.DataFrame(entries, columns=['date', 't_name', 'w', 'l', 'deck', 'player', 'source'])
    return data


def read_links(filename):
    with open(filename) as f:
        urls = [line.strip() for line in f.readlines()]
    return urls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch results from MTGGoldfish tournaments.')
    parser.add_argument('tournaments', action='store', nargs='*', default=[], metavar='URL',
                        help='URLs of tournaments to fetch.')
    parser.add_argument('-r', '--rule-file', action='store', default=None,
                        help='File containing rules to determine deck names based on contents.')
    parser.add_argument('-t', '--tournaments-file', action='store', default=None,
                        help='File containing tournament URLs.')
    arguments = parser.parse_args(sys.argv[1:])
    func = default_name if arguments.rule_file is None else name_from_cards(arguments.rule_file)
    urls = arguments.tournaments if arguments.tournaments_file is None else read_links(arguments.tournaments_file)
    if len(urls) == 0:
        print('Must provide at least one tournament URL.')
        sys.exit(1)
    datasets = [mtggoldfish(url, func) for url in urls]
    data = pandas.concat(datasets)
    data.to_csv(sys.stdout, index=False)
    print(unclassified_cards)
