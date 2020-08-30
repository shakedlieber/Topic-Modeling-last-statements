import bs4
from urllib.request import urlopen
from bs4 import BeautifulSoup
import Prisoner

Prisoners_array = []
prisoner_id = 1
url = "http://www.clarkprosecutor.org/html/death/usexecute.htm"
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')
type(soup)
bs4.BeautifulSoup


# fetch the last words of the prisoners
def get_prisoner_last_words(prisoner_soup):
    all_paragraphs = prisoner_soup.find_all("p")
    for paragraph in all_paragraphs:
        found = paragraph.find("strong")
        if found is not None and ("Last Words:" in found.get_text() or "Final Words:" in found.get_text() or "Final Statement:" in found.get_text() ):
            paragraph_get = paragraph.get_text()
            # if "Last Words:" in paragraph_get or "Final Words:" in paragraph_get:
            last_words = paragraph_get.splitlines()[2:]
            last_words_filtered = filter(lambda x: x != '', last_words)
            all_lines = ''
            for line in last_words_filtered:
                all_lines = all_lines + line
            return all_lines.encode("ascii", "ignore")


# add Prisoner object to Prisoner_array with attributes
def get_prisoner_row(prisoner_soup, prisoner_url):
    global Prisoners_array, prisoner_id
    prisoner_row = prisoner_soup.find("tr", {"valign": "top"})
    all_prisoner_attributes = prisoner_row.find_all("td")
    if all_prisoner_attributes is not None and len(all_prisoner_attributes) == 0:
        return None
    for attribute in all_prisoner_attributes:
        add_spaces = attribute.find_all('br')
        for space in add_spaces:
            space.replaceWith(" ")
        add_spaces = attribute.find_all('p')
        for space in add_spaces:
            space.replaceWith(" "+space.get_text())
        print(attribute.get_text())
    print('--------------------------------------------------')
    prisoner = Prisoner.Prisoner()
    prisoner.constructor_assist(prisoner_id, all_prisoner_attributes)
    prisoner.set_prisoner_url(prisoner_url)
    return prisoner


# get all prisoners form website who have a Link to personal page
def get_all_prisoners_pages():
    global prisoner_id
    all_links = soup.find_all("a")
    # count = 0
    for link in all_links:
        link_get = link.get("href")
        if link_get is not None and "US" in link_get:
            prisoner_page = "http://www.clarkprosecutor.org/html"+link_get[2:]
            prisoner_url = urlopen(prisoner_page)
            prisoner_soup = BeautifulSoup(prisoner_url, 'lxml')
            curr_prisoner = get_prisoner_row(prisoner_soup, prisoner_page)
            if curr_prisoner is None:
                continue
            curr_prisoner.set_last_statement(get_prisoner_last_words(prisoner_soup))
            Prisoners_array.append(curr_prisoner)
            prisoner_id = prisoner_id +1
        #     count = count + 1
        # if count == 5:
        #     break

def scraping_part1():
    # Get the title
    title = soup.title
    get_all_prisoners_pages()
    return Prisoners_array
