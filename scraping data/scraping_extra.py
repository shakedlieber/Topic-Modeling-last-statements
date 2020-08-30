import re

import bs4
from urllib.request import urlopen
from bs4 import BeautifulSoup
import Prisoner
import copy
import urllib

url = "https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html"
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')
type(soup)
bs4.BeautifulSoup
old_array = []
prisoner_id = 0

# fetch the last words of the prisoners
def get_last_statement(pris_page):
    prisoner_page = pris_page
    try:
        prisoner_url = urlopen(prisoner_page)
    except urllib.error.URLError as e:
        print(e.reason)
        return "couldn't fetch last statement"
    prisoner_soup = BeautifulSoup(prisoner_url, 'lxml')
    paragraphs = prisoner_soup.find_all("p")
    i = 0

    for p in paragraphs:
        if re.search(".*Last Statement:.*", p.get_text()):
            try:
                return paragraphs[i+1].get_text()
            except IndexError:
                return "couldn't find last statement"
        i = i + 1
    return "couldn't find last statement"


def find_in_array(last_name, first_name):
    global old_array
    for prisoner in old_array:
        print(prisoner.name)
        if last_name in prisoner.name.decode() and first_name in prisoner.name.decode():
            return prisoner.prisoner_id
    return -1


# get all prisoners form website who have a Link to personal page
def update_array(prisoner_array):
    global prisoner_id
    table_rows = soup.find_all("tr")
    # count = 100
    for row in table_rows:
        print(row)
        row_params = row.find_all("td")
        if len(row_params) > 0:
            prisoner_page = "https://www.tdcj.texas.gov/death_row/" + row_params[2].find("a").get("href")
            index = find_in_array(row_params[3].get_text(), row_params[4].get_text())
            new_last_statement = get_last_statement(prisoner_page)
            if index > 0:
                prisoner_array[index-1].set_last_statement(new_last_statement)
                prisoner_array[index - 1].set_prisoner_url(prisoner_page)
            else:
                new_prisoner = Prisoner.Prisoner(prisoner_id, row_params[7].get_text(), "TX", "NA"
                                                        , row_params[4].get_text()+" "+row_params[3].get_text(), row_params[8].get_text(),
                                                        row_params[7].get_text(), "", row_params[6].get_text(), "", "", "",
                                                        "", "", "", new_last_statement, prisoner_page)
                prisoner_array.append(new_prisoner)
                prisoner_id = prisoner_id + 1
            # count = count - 1
            # if count == 0:
            #     break
    return prisoner_array


def scraping_part2(prisoners_array):
    global old_array
    global prisoner_id
    old_array = copy.deepcopy(prisoners_array)
    prisoner_id = len(prisoners_array)+1
    return update_array(prisoners_array)

