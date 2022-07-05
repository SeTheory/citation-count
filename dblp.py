import argparse
import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import json
import re
from pprint import pprint


# useful attr: _id, abstract, authors, fos, keywords, n_citations, references, title, venue, year
def get_lines_data(data_path):
    with open(data_path + 'dblpv13.json') as fr:
        data = fr.read()
        data = re.sub(r"NumberInt\((\d+)\)", r"\1", data)
        data = json.loads(data)

    info_fw = open(data_path + 'temp_info.lines', 'w+')
    abstract_fw = open(data_path + 'temp_abstract.lines', 'w+')
    ref_fw = open(data_path + 'temp_ref.lines', 'w+')
    for paper in data:
        info_fw.write(str([paper['_id'],
                           {
                               'authors': paper['authors'] if 'authors' in paper else [],
                               'fos': paper['fos'] if 'fos' in paper else [],
                               'keywords': paper['keywords'] if 'keywords' in paper else [],
                               'n_citations': paper['n_citations'] if 'n_citations' in paper else 0,
                               'title': paper['title'] if 'title' in paper else None,
                               'venue': paper['venue'] if 'venue' in paper else None,
                               'year': paper['year'] if 'year' in paper else None,
                           }]) + '\n')
        temp_ab = paper['abstract'] if 'abstract' in paper else None
        temp_ref = paper['references'] if 'references' in paper else []
        abstract_fw.write(str([paper['_id'], temp_ab]) + '\n')
        ref_fw.write(str([paper['_id'], temp_ref]) + '\n')


def get_all_dict(data_path, part='info'):
    result_dict = {}
    with open(data_path + 'temp_{}.lines'.format(part)) as fr:
        for line in fr:
            data = eval(line)
            result_dict[data[0]] = data[1]
    json.dump(result_dict, open(data_path + 'all_{}_dict.json'.format(part), 'w+'))


def get_citation_data(data_path):
    ref_dict = json.load(open(data_path + 'all_ref_dict.json'))
    citation_dict = defaultdict(list)
    for paper in ref_dict:
        # ref_list = ref_dict[paper]
        for ref in ref_dict[paper]:
            citation_dict[ref].append(paper)
    json.dump(citation_dict, open(data_path + 'all_cite_dict.json', 'w+'))



if __name__ == "__main__":
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='Process some description.')
    parser.add_argument('--phase', default='test', help='the function name.')
    parser.add_argument('--data_path', default=None, help='the input.')
    parser.add_argument('--name', default=None, help='file name.')
    # parser.add_argument('--out_path', default=None, help='the output.')
    # parser.add_argument('--seed', default=123, help='the seed.')
    args = parser.parse_args()
    if args.phase == 'test':
        print('This is a test process.')
        # get_pub_data('./data/')
        # get_lines_data('./data/')
        # get_all_dict('./data/', 'info')
        # get_all_dict('./data/', 'abstract')
        # get_all_dict('./data/', 'ref')
        get_citation_data('./data/')
    elif args.phase == 'lines_data':
        get_lines_data(args.data_path)
        print('lines data done')
    elif args.phase == 'info_data':
        get_all_dict(args.data_path, 'info')
        print('lines data done')
    elif args.phase == 'abstract_data':
        get_all_dict(args.data_path, 'abstract')
        print('lines data done')
    elif args.phase == 'ref_data':
        get_all_dict(args.data_path, 'ref')
        print('lines data done')