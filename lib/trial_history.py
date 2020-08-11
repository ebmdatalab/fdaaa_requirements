# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from requests import get
from bs4 import BeautifulSoup
import re
from dateutil.parser import parse
from datetime import date
from time import time

# +
#Gets the html
def get_page(url):
    response = get(url)
    html = response.content
    soup = BeautifulSoup(html, "html.parser")
    return soup

#Grabs the last posted date so we can remove things posted after the date of our full dataset
def check_last_posted(data):
    ss_div = data.find('div', {'id': 'StudyStatusBody'})
    trs = ss_div.find_all('tr')
    for tr in trs:
        if 'Last Update Posted' in tr.text:
            tds = tr.find_all('td')
            last_updated = re.search(r'(\w{3,9} \d{1,2}, \d{4})', tds[-1].text).group(1)
    return parse(last_updated).date()

#ids must be a list of NCT ids
#cutoff is the date of your scrpae
def history_scrape(ids, cutoff):
    docs_list = []
    s_num = 0
    start_time = time() 
    for i in ids:
        s_num += 1
        try:
            url = 'https://clinicaltrials.gov/ct2/history/{}'.format(i)
            soup = get_page(url)
            table = soup.find('table', {'class': 'w3-bordered releases'})
            version_dates = table.find_all('td', {'headers': 'VersionDate'}) #Need this to be able to restrict to changes to specified date
            changes = table.find_all('td', {'headers': 'Changes'})
            if len(changes) == 1:
                pos = 1
                new_url = 'https://clinicaltrials.gov/ct2/history/{0}?A={1}&B={1}&C=merged#StudyPageTop'.format(i, pos)
            else:
                num = 0
                pos = None
                all_doc_changes = []
                for c , v in zip(changes, version_dates):
                    num += 1
                    if 'Documents' in c.text and (parse(v.text).date() <= cutoff):
                        all_doc_changes.append(num)
                        pos = num
                if not pos:
                    pos = 1
                new_url = 'https://clinicaltrials.gov/ct2/history/{0}?A={1}&B={1}&C=merged#StudyPageTop'.format(i, pos)
            version_date = parse(version_dates[pos-1].text).date()
            soup2 = get_page(new_url)
            last_posted = check_last_posted(soup2)
            ind = 2
            if len(all_doc_changes) == 1 and (last_posted > cutoff):
                continue
            else:
                while (last_posted > cutoff) and (ind <= len(all_doc_changes)):
                    new_url = 'https://clinicaltrials.gov/ct2/history/{0}?A={1}&B={1}&C=merged#StudyPageTop'.format(i, all_doc_changes[-ind])
                    soup2 = get_page(new_url)
                    last_posted = check_last_posted(soup2)
                    version_date = parse(version_dates[all_doc_changes[-ind] - 1].text).date()
                    ind += 1              
            documents = soup2.find('div', {'id': 'DocumentsBody'}).find_all(text=True)
            documents = [x for x in documents if x != ' ' and x != '\n']
            no_sap = None
            if 'No Statistical Analysis Plan (SAP) exists for this study.' in documents:
                no_sap = 'No Statistical Analysis Plan (SAP) exists for this study.'
                documents = [x for x in documents if x != no_sap]
            l = len(documents)
            for ix in list(range(0,l,5)):
                tlist = documents[ix:ix+5]
                d = {}
                d['nct_id'] = i
                d['version_date'] = version_date
                d['document_type'] = tlist[0]
                d['document_date'] = re.search(r'(\w{3,9} \d{1,2}, \d{4})', tlist[1]).group(1)
                d['upload_date'] = re.search(r'(\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2})', tlist[2]).group(1)
                if no_sap:
                    d['no_sap'] = no_sap
                docs_list.append(d)
        except Exception as e:
            import sys
            raise type(e)(str(e) +
                          ' happens at {}'.format(i)).with_traceback(sys.exc_info()[2])
    end_time = time()
    print('Completed. {} Trials Scraped of {} in {} Minutes'.format(s_num, len(ids), round((end_time - start_time)/60)))
    return docs_list


# -

