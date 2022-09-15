# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 23:24:41 2018

@author: Gaurdian API
"""

import json
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

ARTICLES_DIR = join('tempdata', 'articles')
makedirs(ARTICLES_DIR, exist_ok=True)
# Sample URL
#
# http://content.guardianapis.com/search?from-date=2016-01-02&
# to-date=2016-01-02&order-by=newest&show-fields=all&page-size=200
# &api-key=your-api-key-goes-here

MY_API_KEY = '40791c78-a8d3-4aa6-807f-0d0e92824b30'
API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
    'from-date': "",
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'section' : 'politics|news|business|uk-news|us-news|global|world',
    'api-key': MY_API_KEY
}


# day iteration from here:
# http://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates
# 2016-1-1 -> 2016-12-31        done !
# 2017-1-1 -> 2018-2-28         done !
# 2018-3-1 -> 2018-9-27         done !

start_date = date(2018, 3, 1)
end_date = date(2018, 9, 27)
dayrange = range((end_date - start_date).days + 1)
fname = join(ARTICLES_DIR, ','.join(my_params['section'].split("|")) + '_' + str((start_date)) + '_' + str((end_date-start_date))[0:-9] + '_.json')
all_results = []
for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')
    if not exists(fname):    
    # then let's download it
        print("Downloading", datestr)
        my_params['from-date'] = datestr
        my_params['to-date'] = datestr
        current_page = 1
        total_pages = 1
        while current_page <= total_pages:
            print("...page", current_page)
            my_params['page'] = current_page
            resp = requests.get(API_ENDPOINT, my_params)
            #print(resp.url)
            data = resp.json()
            all_results.extend(data['response']['results'])
            all_results
            # if there is more than one page
            current_page += 1
            total_pages = data['response']['pages']

with open(fname, 'w') as f:
    print("Writing to", fname)

    # re-serialize it for pretty indentation
    f.write(json.dumps(all_results, indent=2))
    
with open(''.join([fname[0:-5],"total.txt"]), 'w') as f:
    print("Writing total number of results to total.txt")
    
    f.write(str(len(all_results)))