import re
import requests
import sys

import bs4
import pandas


def mtggoldfish(url):
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
            for key, val in t_data.items():
                entry[key] = val
            entries.append(entry)
    data = pandas.DataFrame(entries, columns=['date', 't_name', 'w', 'l', 'deck', 'player', 'source'])
    return data


if __name__ == "__main__":
    urls = sys.argv[1:-1]
    destination = sys.argv[-1]
    datasets = [mtggoldfish(url) for url in urls]
    data = pandas.concat(datasets)
    with open(destination, 'a') as file:
        data.to_csv(file, index=False)
