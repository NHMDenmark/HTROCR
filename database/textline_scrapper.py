import os
import re
import requests
from bs4 import BeautifulSoup as bs
import itertools
import random
import PyPDF2

def collect_danish_texts():
    '''
    Collecting text lines from the following article:
    Madsen, E. 1908. De vigtigste af danske i arktiske
    Egne udførte Rejser og Forskninger. Geografisk
    Tidsskrift. 19, (jan. 1908).
    '''
    url = 'https://tidsskrift.dk/geografisktidsskrift/article/download/49371/63194'
    req = requests.get(url)
    content = req.content
    soup = bs(content.decode('utf-8','ignore'), 'html.parser')
    table_text = soup.find("table").text
    words = table_text.split()
    lines = []
    line = []
    # It is unlikely that museum specimen labels will contain more
    # than 6-7 words per line, hence we restructure line format
    words_per_line = 6
    for word in words:
        if len(line) < words_per_line:
            line.append(word)
        else:
            lines.append(" ".join(line))
            line = [word]
    lines.append(" ".join(line))
    # with open("expeditions_dk.txt", 'w') as w:
    #     w.write(lines)
    return lines

def collect_specie_texts():
    '''
    Collecting text lines in latin based on Plantae kingdom specie
    names. Data collected from gbif api. 
    '''
    url = "https://api.gbif.org/v1/species/search"
    params = {
        "limit": 1000,
        "offset": 1000
    }

    response = requests.get(url, params=params)
    data = response.json()

    # API is either poorly documented or it is not working - search based on kingdom
    # does not return expected output.
    plantae_species = [el['scientificName'] for el in data['results'] if el['kingdom'] == 'Plantae']

    # Using combinations of extracted entries
    def generate_combinations(entries, size):
        return list(map(lambda x: ' '.join(x), itertools.combinations(entries, size)))

    species2 = plantae_species[:100]
    species3 = plantae_species[100:175]
    species4 = plantae_species[175:200]
    # Generating combinations in different lengths
    combinations2 = generate_combinations(species2, 2)
    combinations3 = generate_combinations(species3, 3)
    combinations4 = generate_combinations(species4, 4)
    comb = combinations2 + combinations3 + combinations4
    random.shuffle(comb)
    with open("gbif.txt", 'w') as w:
        w.write('\n'.join(line for line in comb))        


def collect_english_texts():
    '''
    Collecting text lines from the following articles:

    Dorte Bugge Jensen. The Biodiversity of Greenland – a country study
    Safi-Kristine Darden - English translation
    https://natur.gl/wp-content/uploads/2019/07/55-Biodiversity_of_Greenland.pdf

    Rasmussen, K., Ostenfeld, C. H., Porsild, M. P., & Koch, L. (1919).
    Scientific Results of the Second Thule Expedition to Northern Greenland,
    1916-1918. Geographical Review, 8(3), 180–187. https://doi.org/10.2307/207406

    Rasmussen, Knud. “The Second Thule Expedition to Northern Greenland,
    1916-1918.” Geographical Review, vol. 8, no. 2, 1919, pp. 116–25. 
    JSTOR, https://doi.org/10.2307/207633. 

    Jenness, Diamond. “A New Eskimo Culture in Hudson Bay.” 
    Geographical Review, vol. 15, no. 3, 1925, pp. 428–37. JSTOR, https://doi.org/10.2307/208564.

    Knud Rasmussen, et al. “The Danish Ethnographic and Geographic Expedition to Arctic America.
    Preliminary Report of the Fifth Thule Expedition.” Geographical Review, vol. 15, no. 4, 1925, pp. 521–62.
    JSTOR, https://doi.org/10.2307/208623. 

    William Hovgaard. “The Norsemen in Greenland: Recent Discoveries at Herjolfsnes.”
    Geographical Review, vol. 15, no. 4, 1925, pp. 605–16. JSTOR, https://doi.org/10.2307/208626. 
    '''
    article_list = ['55-Biodiversity_of_Greenland.pdf']#, '207406.pdf', '207633.pdf', '208564.pdf', '208623.pdf', '208626.pdf']
    lines = []
    for article in article_list:
        with open(os.path.join("articles", article), "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            num_pages = len(pdf.pages)
            text = ""
            for i in range(num_pages):
                page = pdf.pages[i]
                text += page.extract_text()

            text = " ".join(text.split())
            # Remove meta info with unknown unicode characters 
            regex = r'This content downloaded from .*? subject to https:\/\/about\.jstor\.org\/terms'
            text = re.sub(regex, '', text)
            # Remove any links
            regex = r'https:\/\/.*?[\t\r\n\s]'
            text = re.sub(regex, '', text)
            # Remove ToC rows
            regex = r'\.{4,}'
            text = re.sub(regex, '', text)
            # Remove super long words (error of pdf reader)
            regex = r'\b\w{45,}\b'
            text = re.sub(regex, '', text)
            words = text.split()
            line = []
            words_per_line = random.randint(3,8)
            counter = 0
            for word in words:
                counter += 1
                if counter == 100:
                    words_per_line = random.randint(3,8)
                    counter = 0
                
                if len(line) < words_per_line:
                    line.append(word)
                else:
                    lines.append(" ".join(line))
                    line = [word]
            lines.append(" ".join(line))
            collection = "\n".join(lines)
    with open("texts_en.txt", 'w') as w:
        w.write(collection)
    return lines

def collect_location_info(n=10000):
    '''
    Generating a large number of latitude, longitude coordinates.
    '''
    coordinates = []
    for _ in range(n):
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)
        latitude_dir = "N" if latitude >= 0 else "S"
        longitude_dir = "E" if longitude >= 0 else "W"
        latitude = abs(latitude)
        longitude = abs(longitude)
        latitude_deg = int(latitude)
        latitude_min = int((latitude - latitude_deg) * 60)
        longitude_deg = int(longitude)
        longitude_min = int((longitude - longitude_deg) * 60)
        coordinates.append("{}° {}'".format(
            latitude_deg, latitude_min))
    with open("coords_short.txt", 'w') as w:
            w.write('\n'.join(coord for coord in coordinates))
    return coordinates


# collect_danish_texts()
# collect_english_texts()
collect_location_info()
# collect_specie_texts()