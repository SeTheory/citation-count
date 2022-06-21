import argparse
import datetime
import os
import pandas as pd
from bs4 import BeautifulSoup, Tag
import json

def get_valid_ref_list(soup):
    result_dict = {}
    ref_list = soup.find('ref-list')
    if ref_list:
        ref_list = ref_list.find_all('ref')
        for ref in ref_list:
            ref_label = ref.attrs['id']
            ref_ids = ref.find_all('pub-id')
            if len(ref_ids) < 1:
                continue
    #         print(ref_ids)
            result_dict[ref_label] = {ref_id.attrs['pub-id-type']: ref_id.text  for ref_id in ref_ids}
    return result_dict


def get_article_meta(soup):
    article = soup.front.find('article-meta')
    if article.find('kwd-group'):
        kwds = [kwd.text for kwd in article.find('kwd-group').find_all('kwd')]
    else:
        kwds = []

    if article.find('abstract'):
        abstract = article.find('abstract').text  # 没清理过的latex内容，需要进一步处理
    else:
        abstract = ''

    pub_date = {'year': article.find('pub-date').find('year').text if article.find('pub-date').find('year') else None,
                'month': article.find('pub-date').find('month').text if article.find('pub-date').find(
                    'month') else None,
                'day': article.find('pub-date').find('day').text if article.find('pub-date').find('day') else None}

    if article.find('contrib-group'):
        authors = article.find('contrib-group').find_all('contrib')
        if authors:
            authors = [{'type': contrib.attrs, 'name': {name.name: name.text for name in contrib.find('name').children
                                                        if isinstance(name, Tag)}}
                       for contrib in authors if contrib.find('name')]
        else:
            authors = []
        aff = [str(item) for item in article.find('contrib-group').find_all('aff')]  # 没想好怎么处理机构
    else:
        authors = []
        aff = []

    cat = article.find('article-categories').subject.text if article.find('article-categories') else None
    ids = {arc_id.attrs['pub-id-type']: arc_id.text for arc_id in article.find_all('article-id')}
    title = article.find('article-title').text if article.find('article-title') else None

    journal = soup.front.find('journal-meta')
    if journal:
        journal_ids = {jid.attrs['journal-id-type']: jid.text for jid in journal.find_all('journal-id')}
        journal_title = journal.find('journal-title').text if journal.find('journal-title') else None
    else:
        journal_ids = {}
        journal_title = None

    if soup.body:
        full = len(soup.body.find_all('xref', attrs={'ref-type': 'bibr'})) > 0
    else:
        full = False

    return {
        'pmc': ids['pmc'],
        'ids': ids,
        'title': title,
        'cat': cat,
        'aff': aff,
        'authors': authors,
        'pub_date': pub_date,
        'abstract': abstract,
        'kwds': kwds,
        'journal': {'ids': journal_ids, 'title': journal_title},
        'full': full
    }


def get_simple_data(data_path, name):
    ref_dict = {}
    pub_dict = {}
    folder_list = os.listdir(data_path)
    for folder in folder_list:
        folder_path = data_path + folder
        files = os.listdir(folder_path)
        for file in files:
            file_path = folder_path + '/' + file
            #         print(file_path)
            try:
                with open(file_path) as fr:
                    sample = fr.readlines()
                soup = BeautifulSoup(''.join(sample), ["lxml-xml"])

                temp_ref = get_valid_ref_list(soup)
                temp_pub = get_article_meta(soup)
                cur_pmc = temp_pub['pmc']
                ref_dict[cur_pmc] = temp_ref
                pub_dict[cur_pmc] = temp_pub
            except Exception as e:
                print(e)
                print(file_path)
            finally:
                continue
    json.dump(ref_dict, open('./data/{}.ref'.format(name), 'w+', encoding='utf-8'))
    json.dump(pub_dict, open('./data/{}.pub'.format(name), 'w+', encoding='utf-8'))


def get_lines_data(data_path, name):
    folder_list = os.listdir(data_path)
    ref_fr = open('./data/{}_lines.ref'.format(name), 'w+', encoding='utf-8')
    pub_fr = open('./data/{}_lines.pub'.format(name), 'w+', encoding='utf-8')
    count = 0
    for folder in folder_list:
        folder_path = data_path + folder
        files = os.listdir(folder_path)
        for file in files:
            file_path = folder_path + '/' + file
            #         print(file_path)
            count += 1
            try:
                with open(file_path) as fr:
                    sample = fr.readlines()
                soup = BeautifulSoup(''.join(sample), ["lxml-xml"])
                temp_ref = get_valid_ref_list(soup)
                temp_pub = get_article_meta(soup)
                cur_pmc = temp_pub['pmc']
                ref_fr.write(str({'pmc': cur_pmc, 'ref': temp_ref}) + '\n')
                pub_fr.write(str({'pmc': cur_pmc, 'pub': temp_pub}) + '\n')
            except Exception as e:
                print(e)
                print(file_path)
            finally:
                continue

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
    elif args.phase == 'simple_data':
        get_simple_data(args.data_path, args.name)
        print('simple data done')
    elif args.phase == 'lines_data':
        get_lines_data(args.data_path, args.name)
        print('lines data done')