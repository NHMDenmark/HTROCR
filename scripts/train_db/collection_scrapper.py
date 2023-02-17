import os
import re
import requests
from lxml import etree
from bs4 import BeautifulSoup as bs

TRAIN_DB_DIR = '../../data/training_data'

def collect_emunch_db(notebook):
    '''
    Given the EMunch notebook title, the function collects a list of pages 
    and downloads page-level images along with coresponding transcriptions.
    '''

    URL = 'https://emunch.no/TEXT{}.xhtml'.format(notebook)

    req = requests.get(URL)
    content = req.content
    soup = bs(content.decode('utf-8','ignore'), 'html.parser')

    # Page name tags
    page_ids = soup.find_all("div", {"class": "pageID"})
    pages_to_download = [pi.string for pi in page_ids]

    # Creating databases 
    labels_dir = os.path.join(TRAIN_DB_DIR, 'emunch', notebook, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    images_dir = os.path.join(TRAIN_DB_DIR, 'emunch', notebook, 'images')
    os.makedirs(images_dir, exist_ok=True)


    def attempt_download(filename):
        image_url = 'https://emunch.no/T/full/{}'.format(filename)
        return requests.get(image_url).content

    for i in range(len(page_ids)):
        current_div = page_ids[i]
        if i != len(page_ids) - 1:
            next_div = page_ids[i + 1]

        span_texts = []

        # Follow https://emunch.no transcription format
        for element in current_div.find_next_siblings():
            # for the last element - read until any `div` is found
            if element == next_div or (i == len(page_ids) - 1 and element.name == 'div'):
                break

            span_texts.append(element.text)

        # filter out empty rows
        span_texts = [i for i in span_texts if i]
        transcription = "\n".join(span_texts)

        # If there are transcriptions present - proceed.
        if transcription.strip() != '':
            # Transform filename to the emunch format:
            words = pages_to_download[i].split()
            if len(words) > 1:
                page_no = words[-1] if len(words[-1]) == 3 else '0' + words[-1]
                file = '{}-{}'.format(notebook, page_no)
            else:
                file = words[0]

            # Attempt to download the image
            filename = file + '.jpg'
            image_data = attempt_download(filename)
            # Filename does not follow the standard anymore - adjust the name
            # and try again. If failed - ignore.
            if len(image_data) < 200000:
                page_no = words[-1]
                file = '{}-{}'.format(notebook, page_no)
                filename = file + '.jpg'
                image_data = attempt_download(filename)

            # Attempt to change page number format again
            if len(image_data) < 200000 and len(words[-1]) == 3:
                page_no = '0' + words[-1]
                file = '{}-{}'.format(notebook, page_no)
                filename = file + '.jpg'
                image_data = attempt_download(filename)

            # This time - skip
            if len(image_data) < 200000:
                break

            image_loc = os.path.join(images_dir, filename)
            with open(image_loc, 'wb') as wb:
                wb.write(image_data)
            label_loc = os.path.join(labels_dir,  file + ".txt")
            with open(label_loc, 'w') as w:
                w.write(transcription)

def collect_mmd_db(notebook_info):
    '''
    Given the start page, end page and collection (record) id, the function collects a list of pages 
    and downloads page-level images along with coresponding transcriptions. 
    NOTE: Transcriptions have no line breaks.
    '''

    labels_dir = os.path.join(TRAIN_DB_DIR, 'mmd', str(notebook_info[3]), 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    images_dir = os.path.join(TRAIN_DB_DIR, 'mmd', str(notebook_info[3]), 'images')
    os.makedirs(images_dir, exist_ok=True)
    for id in range(notebook_info[0], notebook_info[1] + 1):
        URL = 'https://contentdm.lib.byu.edu/digital/collection/MMD/id/{}/rec/{}'.format(id, notebook_info[2])
        req = requests.get(URL)
        content = req.content
        soup = bs(content.decode('utf-8','ignore'), 'html.parser')
        text = str(soup)

        # Website does not work properly without javascript, but it still provides
        # transcriptions within a script tag as a dense stream of text.
        # Using regex to extract only the transcription part.
        regex = '\\\\"Page_\\\\",\\\\"text\\\\":\\\\"(.+)\\\\",\\\\"pageNumber\\\\"'
        x = re.findall(regex, text)
        if len(x) == 0:
            continue

        transcription = x[0].replace("\\", '')

        # Download image
        image_url = 'https://contentdm.lib.byu.edu/digital/iiif/MMD/{}/full/full/0/default.jpg'.format(id)
        image_data = requests.get(image_url).content
        filename = str(id) + '.jpg'
        # Write image and its transcription
        image_loc = os.path.join(images_dir, filename)
        with open(image_loc, 'wb') as wb:
            wb.write(image_data)
        label_loc = os.path.join(labels_dir,  str(id) + ".txt")
        with open(label_loc, 'w') as w:
            w.write(transcription)


def collect_ibsen_db(period):
    '''
    Given the year period, the function collects records from web
    and downloads page-level images. Transcriptions need to be collected manually
    since 
    '''
    regex = "\.open\('(.+)', 'facsimileVindu"
    sample_counter = 0
    for year in range(period[0], period[1], 1):
        # First we need to get a list of documents per year
        catalog_url = 'https://www.ibsen.uio.no/brevOversikt_{}.xhtml'.format(year)
        req = requests.get(catalog_url)
        content = req.content
        soup = bs(content.decode('utf-8','ignore'), 'html.parser')
        # Links to documents are positioned in table
        links = soup.select("td > a")
        letter_page_urls = ['https://www.ibsen.uio.no/' + link['href'] for link in links]
        labels_dir = os.path.join(TRAIN_DB_DIR, 'ibsen', str(year), 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        images_dir = os.path.join(TRAIN_DB_DIR, 'ibsen', str(year), 'images')
        os.makedirs(images_dir, exist_ok=True)
        # Go through each letter and collect information
        for lp_url in letter_page_urls:
            lp_req = requests.get(lp_url)
            lp_content = lp_req.content
            lp_soup = bs(lp_content.decode('utf-8','ignore'), 'html.parser')
            link_tags = lp_soup.select("td > div > img")
            for l in link_tags:
                text = l['onclick']
                r = re.findall(regex, text)
                image_url = r[0]
                image_data = requests.get(image_url).content
                # Image link is broken in their website
                if len(image_data) < 500:
                    continue

                filename = "sample_{}".format(sample_counter)
                sample_counter += 1
                # Write image and create transcription file for manual insertion
                image_loc = os.path.join(images_dir, filename + ".jpg")
                with open(image_loc, 'wb') as wb:
                    wb.write(image_data)
                label_loc = os.path.join(labels_dir,  filename + ".txt")
                with open(label_loc, 'w') as w:
                    pass
        


if __name__ == '__main__':
    # Manually refined collection, that contains text in the notebook pages
    emunch_collection = \
    [
        "No-MM_T2704", "No-MM_T2734", "No-MM_T2748",
        "No-MM_T2759", "No-MM_T2760", "No-MM_T2761",
        "No-MM_T2770", "No-MM_T2771", "No-MM_T2786",
        "No-MM_T2789","No-MM_T2794", "No-MM_T2797",
        "No-MM_T2800","No-MM_T2892","No-MM_T2893",
        "No-MM_T2924"
    ]

    for notebook in emunch_collection:
        collect_emunch_db(notebook)

    # 10 notebooks
    # Format: [Starting id, Ending id, Record id, Year]
    mmd_collection = \
    [
        [29892, 29983, 1, 1881],  # Thatcher, Moses (Mexico) vol. 3, 1881
        [44428, 44527, 14, 1922], # Taylor, George Shepherd vol. 08, 1922-1923
        [54201, 54339, 17, 1852], # Harper, Charles Alfred vol. 1, 1852
        [41773, 41927, 22, 1901], # Erekson, William Benbow vol. 5, 1901
        [59797, 59864, 38, 1855], # Jones, Samuel Stephen vol. 1, 1855
        [60389, 60458, 67, 1865], # Hatch, Abram vol. 04, 1865
        [66241, 66388, 73, 1900], # Nielson, Frihoff G. vol. 2, 1900-1901
        [22252, 22423, 117, 1910], # Ivie, Lloyd Oscar vol. 1, 1910-1911
        [50238, 50320, 173, 1847], # Huntington, Oliver Boardman book 7, 1847
        [59044, 59360, 189, 1867], # Thatcher, Moses (British Isles) vol. 2, 1867-1868
        
    ]

    for notebook in mmd_collection:
        collect_mmd_db(notebook)


    # Format: [Start year, End year]
    ibsen_collection = \
    [
        [1850, 1860],
        [1860, 1870],
        [1870, 1880],
        [1880, 1890],
        [1890, 1900],
        [1900, 1906],
    ]

    for period in ibsen_collection:
        collect_ibsen_db(period)