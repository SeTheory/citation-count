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
    ref_fr = open('./data/pubmed/{}_lines.ref'.format(name), 'w+', encoding='utf-8')
    pub_fr = open('./data/pubmed/{}_lines.pub'.format(name), 'w+', encoding='utf-8')
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
                temp_pub['file_path'] = file_path
                cur_pmc = temp_pub['pmc']
                ref_fr.write(str({'pmc': cur_pmc, 'ref': temp_ref}) + '\n')
                pub_fr.write(str({'pmc': cur_pmc, 'pub': temp_pub}) + '\n')
            except Exception as e:
                print(e)
                print(file_path)
            finally:
                continue


def get_pub_data(data_path):
    pub_dict = {}
    files_list = os.listdir(data_path)
    files_list = [file for file in files_list if file.endswith('lines.pub')]
    for file in files_list:
        print(file)
        with open(data_path + file, 'r', encoding='utf-8') as fr:
            temp_data = list(map(lambda x: eval(x), fr.readlines()))
        for paper in temp_data:
            pub_dict[paper['pmc']] = paper['pub']
    json.dump(pub_dict, open(data_path + 'all_pub_data.json', 'w+'))

    doi_trans = {}
    pmid_trans = {}
    for paper in pub_dict.values():
        if 'pmid' in paper['ids']:
            pmid_trans[paper['ids']['pmid']] = paper['pmc']
        if 'doi' in paper['ids']:
            doi_trans[paper['ids']['doi']] = paper['pmc']
    json.dump(pmid_trans, open(data_path + 'pmid_trans.json', 'w+'))
    json.dump(doi_trans, open(data_path + 'doi_trans.json', 'w+'))

    with open(data_path + 'all_ids.list', 'w+') as fw:
        fw.writelines([pmc + '\n' for pmc in pub_dict])


def get_ref_data(data_path):
    ref_dict = {}
    files_list = os.listdir(data_path)
    files_list = [file for file in files_list if file.endswith('lines.ref')]
    doi_trans = json.load(open(data_path + 'doi_trans.json'))
    pmid_trans = json.load(open(data_path + 'pmid_trans.json'))

    print('loading data...')
    for file in files_list:
        print(file)
        with open(data_path + file, 'r', encoding='utf-8') as fr:
            temp_data = list(map(lambda x: eval(x), fr.readlines()))
        for paper in temp_data:
            ref_dict[paper['pmc']] = paper['ref']

    dealt_ref = {}
    print('dealting data...')
    with open(data_path + 'all_ids.list', 'r+') as fr:
        all_paper_pmc = list(map(lambda x: x.strip(), fr.readlines()))
    for paper in ref_dict:
        ref_list = []
        temp_ref = ref_dict[paper].values()
        for ref_paper in temp_ref:
            if 'pmcid' in ref_paper:
                if ref_paper['pmcid'] in all_paper_pmc:
                    ref_list.append(ref_paper['pmcid'])
            elif 'pmid' in ref_paper:
                if ref_paper['pmid'] in pmid_trans:
                    ref_list.append(pmid_trans[ref_paper['pmid']])
            elif 'doi' in ref_paper:
                if ref_paper['doi'] in doi_trans:
                    ref_list.append(doi_trans[ref_paper['doi']])
        dealt_ref[paper] = ref_list

    json.dump(dealt_ref, open(data_path + 'all_ref_data.json', 'w+'))

    print('dealting citing data...')
    cite_dict = {paper: [] for paper in all_paper_pmc}
    for (paper, ref_list) in dealt_ref.items():
        for ref_paper in ref_list:
            cite_dict[ref_paper].append(paper)
    json.dump(cite_dict, open(data_path + 'all_cite_data.json', 'w+'))

    del ref_dict

    year_dict = {}
    pub_dict = json.load(open(data_path + 'all_pub_data.json', 'r'))
    for paper in cite_dict:
        year_dict[paper] = list(map(lambda x: int(pub_dict[x]['pub_date']['year']), cite_dict[paper]))
    json.dump(year_dict, open(data_path + 'cite_year_data.json', 'w+'))


def show_data(data_path):
    data = json.load(open(data_path + 'all_pub_data.json', 'r'))
    print(list(data.values())[0])
    print(list(data.values())[0].keys())
    data = map(lambda x: {
        'pmc': x['pmc'],
        'type': x['cat'],
        'aff': len(x['aff']),
        'authors': len(x['authors']),
        'pub_date': int(x['pub_date']['year']),
        'abstract': len(x['abstract'].split(' ')) >= 20,
        'kwds': len(x['kwds']),
        'journal': len(x['journal']['ids']) > 0,
        'full': x['full']
    }, data.values())

    df = pd.DataFrame(data)
    print(df.shape)
    print(df.columns)
    print(df.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]))
    print(df.groupby('type').count().sort_values(by='pmc', ascending=False)['pmc'])
    df.groupby('type').count()['pmc'].to_csv(data_path + 'type_count.csv')
    print(df.groupby('pub_date').count().sort_values(by='pmc', ascending=False)['pmc'])
    df.groupby('pub_date').count()['pmc'].to_csv(data_path + 'pub_date_count.csv')
    print(df.groupby('abstract').count().sort_values(by='pmc', ascending=False)['pmc'])
    print(df.groupby('full').count().sort_values(by='pmc', ascending=False)['pmc'])


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
        show_data('./data/')
    elif args.phase == 'simple_data':
        get_simple_data(args.data_path, args.name)
        print('simple data done')
    elif args.phase == 'lines_data':
        get_lines_data(args.data_path, args.name)
        print('lines data done')
    elif args.phase == 'pub_data':
        get_pub_data(args.data_path)
        print('pub data done')
    elif args.phase == 'ref_data':
        get_ref_data(args.data_path)
        print('ref data done')
    elif args.phase == 'show_data':
        show_data(args.data_path)
        print('show data done')