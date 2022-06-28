import argparse
import datetime
import json
import os

import pandas as pd
from collections import defaultdict


def reorganize_data(data_path):
    files = os.listdir(data_path + 'metadata/')
    info_fw = open(data_path + 'all_info.lines', 'w+')
    ab_fw = open(data_path + 'all_abstract.lines', 'w+')
    citation_fw = open(data_path + 'all_cite.lines', 'w+')
    reference_fw = open(data_path + 'all_ref.lines', 'w+')
    for file in files:
        with open(data_path + 'metadata/' + file) as fr:
            print(file)
            for line in fr:
                temp_data = json.loads(line)
                temp_info = {
                    'paper_id': temp_data['paper_id'],
                    'title': temp_data['title'],
                    'authors': temp_data['authors'],
                    'year': temp_data['year'],
                    'journal': temp_data['journal'],
                    'venue': temp_data['venue'],
                    'mag_field_of_study': temp_data['mag_field_of_study'],
                    'citations': len(temp_data['inbound_citations']),
                    'has_pdf_parse': temp_data['has_pdf_parse'],
                    'file_name': file
                }
                info_fw.write(str(temp_info) + '\n')
                ab_fw.write(str([temp_data['paper_id'], temp_data['abstract']]) + '\n')
                citation_fw.write(str([temp_data['paper_id'], temp_data['inbound_citations']]) + '\n')
                reference_fw.write(str([temp_data['paper_id'], temp_data['outbound_citations']]) + '\n')
    info_fw.close()
    ab_fw.close()
    citation_fw.close()
    reference_fw.close()


def get_info_dict(data_path):
    info_dict = {}
    with open(data_path + 'all_info.lines') as fr:
        for line in fr:
            temp_data = eval(line)
            info_dict[temp_data['paper_id']] = temp_data
    json.dump(info_dict, open(data_path + 'all_info_dict.json', 'w+'))


def get_cite_dict(data_path):
    cite_dict = {}
    with open(data_path + 'all_cite.lines') as fr:
        for line in fr:
            temp_data = eval(line)
            cite_dict[temp_data[0]] = temp_data[1]
    json.dump(cite_dict, open(data_path + 'all_cite_dict.json', 'w+'))


def cite_year_count(data_path):
    year_dict = {}
    cite_year_dict = {}
    # info_dict = json.load(open(data_path + 'all_info_dict.json', 'r'))
    with open(data_path + 'all_info.lines') as fr:
        for line in fr:
            temp_data = eval(line)
            year_dict[temp_data['paper_id']] = temp_data['year']
    json.dump(year_dict, open(data_path + 'all_year_dict.json', 'w+'))

    # for paper in cite_dict:
    #     cur_valid = [paper for paper in cite_dict[paper] if paper in info_dict]
    #     year_dict[paper] = list(filter(lambda x: x, map(lambda x: info_dict[x]['year'], cur_valid)))
    cite_dict = json.load(open(data_path + 'all_cite_dict.json', 'r'))
    for paper in cite_dict:
        cur_valid = [paper for paper in cite_dict[paper] if paper in year_dict]
        cite_year_dict[paper] = list(filter(lambda x: x, map(lambda x: year_dict[x], cur_valid)))
    json.dump(cite_year_dict, open(data_path + 'all_cite_year.json', 'w+'))


def get_split_info_dict(data_path):
    temp_title_fw = open(data_path + 'temp_title.lines', 'w+')
    temp_authors_fw = open(data_path + 'temp_authors.lines', 'w+')
    temp_source_fw = open(data_path + 'temp_source.lines', 'w+')
    temp_cat_fw = open(data_path + 'temp_cat.lines', 'w+')
    temp_citation_count_fw = open(data_path + 'temp_citation_count.lines', 'w+')
    temp_pdf_fw = open(data_path + 'temp_pdf.lines', 'w+')
    with open(data_path + 'all_info.lines') as fr:
        for line in fr:
            temp_data = eval(line)
            temp_title_fw.write(str({'id': temp_data['paper_id'], 'values': temp_data['title']}) + '\n')
            temp_authors_fw.write(str({'id': temp_data['paper_id'], 'values': temp_data['authors']}) + '\n')
            temp_source_fw.write(str({'id': temp_data['paper_id'],
                                      'values': {'journal': temp_data['journal'],
                                                 'venue': temp_data['venue']
                                                 }}) + '\n')
            temp_cat_fw.write(str({'id': temp_data['paper_id'],
                                   'values': temp_data['mag_field_of_study']}) + '\n')
            temp_citation_count_fw.write(str({'id': temp_data['paper_id'], 'values': temp_data['citations']}) + '\n')
            temp_pdf_fw.write(str({'id': temp_data['paper_id'],
                                   'values': {
                                       'has_pdf_parse': temp_data['has_pdf_parse'],
                                       'file_name': temp_data['file_name']
                                   }}) + '\n')
    temp_title_fw.close()
    temp_authors_fw.close()
    temp_source_fw.close()
    temp_cat_fw.close()
    temp_citation_count_fw.close()
    temp_pdf_fw.close()

    files = ['title', 'authors', 'source', 'cat', 'citation_count', 'pdf']
    for file in files:
        temp_dict = {}
        with open(data_path + 'temp_{}.lines'.format(file)) as fr:
            for line in fr:
                temp_data = eval(line)
                # print(temp_data)
                temp_dict[temp_data['id']] = temp_data['values']
        json.dump(temp_dict, open(data_path + 'all_{}_dict.json'.format(file), 'w+'))


# def show_data(data_path):
#     # result_dict = {}
#
#     authors_dict = json.load(open(data_path + 'all_authors_dict.json', 'r'))
#     result_dict = {paper_id: {'authors': len(authors_list) if isinstance(authors_list, list) else 0}
#                    for (paper_id, authors_list) in authors_dict.items()}
#     del authors_dict
#     print('authors done')
#
#     pdf_dict = json.load(open(data_path + 'all_pdf_dict.json', 'r'))
#     # pdf_values = {paper_id: values['has_pdf_parse'] for (paper_id, values) in pdf_dict.item()}
#     for (paper_id, values) in pdf_dict.items():
#         result_dict[paper_id]['full'] = values['has_pdf_parse']
#     del pdf_dict
#     print('pdf done')
#
#     cat_dict = json.load(open(data_path + 'all_cat_dict.json', 'r'))
#     # cat_values = {paper_id: len(cat_list) for (paper_id, cat_list) in cat_dict.items()}
#     for (paper_id, cat_list) in cat_dict.items():
#         result_dict[paper_id]['cat'] = len(cat_list) if isinstance(cat_list, list) else 0
#     del cat_dict
#     print('cat done')
#
#     citation_dict = json.load(open(data_path + 'all_citation_count_dict.json', 'r'))
#     for (paper_id, value) in citation_dict.items():
#         result_dict[paper_id]['citation'] = value
#     print('citation done')
#
#     year_dict = json.load(open(data_path + 'all_year_dict.json', 'r'))
#     for (paper_id, value) in year_dict.items():
#         result_dict[paper_id]['year'] = value
#     print('year done')
#
#     df = pd.DataFrame(result_dict)
#     print(df.describe())
#
#     print(df.groupby('year').count())
#     df.groupby('year').count().to_csv(data_path + 'year_count.csv')
#     print(df.groupby('full').count())
#
#     del df, result_dict
def get_count_detail(input_dict, max_value, file_name, sort_by_count=False):
    df = pd.DataFrame.from_dict(input_dict, orient='index')
    df.columns = ['count']
    if sort_by_count:
        df = df.sort_values(by='count', ascending=False)
    else:
        df = df.sort_index()
    df['count_cum'] = df['count'].cumsum()
    df['percent'] = df['count'] / max_value
    df['percent_cum'] = df['count_cum'] / max_value
    print(df.head(10))
    df.to_csv(file_name)


def show_data(data_path):
    # 暂时去除作者
    # count_dict = {}
    # authors_dict = json.load(open(data_path + 'all_authors_dict.json', 'r'))
    # count_sum = len(authors_dict)
    # for authors_list in authors_dict.values():
    #     count = len(authors_list) if isinstance(authors_list, list) else 0
    #     if count in count_dict:
    #         count_dict[count] += 1
    #     else:
    #         count_dict[count] = 1
    # del authors_dict
    # get_count_detail(count_dict, count_sum, data_path + 'authors_count.csv')
    # print('authors done')

    count_dict = {}
    pdf_dict = json.load(open(data_path + 'all_pdf_dict.json', 'r'))
    # pdf_values = {paper_id: values['has_pdf_parse'] for (paper_id, values) in pdf_dict.item()}
    count_dict = {0: 0, 1: 0}
    for values in pdf_dict.values():
        count_dict[int(values['has_pdf_parse'])] += 1
    print(count_dict)
    del pdf_dict
    print('pdf done')
    #
    count_dict = {}
    cat_dict = json.load(open(data_path + 'all_cat_dict.json', 'r'))
    # cat_values = {paper_id: len(cat_list) for (paper_id, cat_list) in cat_dict.items()}
    cat_count_dict = {}
    count_sum = len(cat_dict)
    for cat_list in cat_dict.values():
        count = len(cat_list) if isinstance(cat_list, list) else 0
        if count in count_dict:
            count_dict[count] += 1
        else:
            count_dict[count] = 1

        if isinstance(cat_list, list):
            for cat in cat_list:
                if cat in cat_count_dict:
                    cat_count_dict[cat] += 1
                else:
                    cat_count_dict[cat] = 1
    get_count_detail(count_dict, count_sum, data_path + 'cat_count.csv')
    get_count_detail(cat_count_dict, count_sum, data_path + 'cat_type_count.csv', True)
    del cat_dict
    del cat_count_dict
    print('cat done')
    #
    count_dict = {}
    citation_dict = json.load(open(data_path + 'all_citation_count_dict.json', 'r'))
    count_sum = len(citation_dict)
    for value in citation_dict.values():
        count = value
        if count in count_dict:
            count_dict[count] += 1
        else:
            count_dict[count] = 1
    get_count_detail(count_dict, count_sum, data_path + 'citation_count.csv')
    del citation_dict
    print('citation done')
    #
    count_dict = {}
    year_dict = json.load(open(data_path + 'all_year_dict.json', 'r'))
    count_sum = len(year_dict)
    for value in year_dict.values():
        count = value
        if count in count_dict:
            count_dict[count] += 1
        else:
            count_dict[count] = 1
    get_count_detail(count_dict, count_sum, data_path + 'year_count.csv')
    print('year done')
    del year_dict

    count_dict = {}
    journal_dict = {}
    venue_dict = {}
    source_dict = json.load(open(data_path + 'all_source_dict.json', 'r'))
    count_sum = len(source_dict)
    for values in source_dict.values():
        journal = values['journal']
        if journal in journal_dict:
            journal_dict[journal] += 1
        else:
            journal_dict[journal] = 1

        venue = values['venue']
        if venue in venue_dict:
            venue_dict[venue] += 1
        else:
            venue_dict[venue] = 1
    del source_dict
    get_count_detail(journal_dict, count_sum, data_path + 'journal_count.csv', True)
    get_count_detail(venue_dict, count_sum, data_path + 'venue_count.csv', True)


def show_abstract_stats(data_path):
    # count = 0
    count_dict = {0: 0, 1: 1}
    with open(data_path + 'all_abstract.lines', 'r') as fr:
        for line in fr:
            temp_abstract = eval(line)[1]
            if temp_abstract:
                ab_length = len(temp_abstract.split(' '))
            else:
                ab_length = 0
            if ab_length >= 20:
                count_dict[1] += 1
            else:
                count_dict[0] += 1
            # count += 1
    print(count_dict)


def show_selected_stats(data_path, time_range):
    files = os.listdir(data_path + 'metadata/')
    all_count = 0
    pdf_dict = {0: 0, 1: 0}
    cat_dict = defaultdict(int)
    citation_dict = defaultdict(int)
    title_dict = {0: 0, 1: 0}
    abstract_dict = {0: 0, 1: 0}
    authors_dict = defaultdict(int)
    journal_dict = defaultdict(int)
    venue_dict = defaultdict(int)
    # inbound_dict = defaultdict(int)
    ref_dict = defaultdict(int)
    good_paper_dict = {0: 0, 1: 0}

    for file in files:
        with open(data_path + 'metadata/' + file) as fr:
            print(file)
            for line in fr:
                temp_data = json.loads(line)
                year = temp_data['year'] if temp_data['year'] else 0
                if (year >= time_range[0]) & (year < time_range[1]):
                    all_count += 1
                    # pdf
                    pdf_dict[int(temp_data['has_pdf_parse'])] += 1
                    # authors
                    authors_dict[len(temp_data['authors'])] += 1
                    # title
                    useful_title = True if temp_data['title'] else False
                    title_dict[int(useful_title)] += 1
                    # abstract
                    useful_abstract = len(temp_data['abstract'].split(' ')) >= 20 if temp_data['abstract'] else False
                    abstract_dict[int(useful_abstract)] += 1
                    # cat
                    cat_list = temp_data['mag_field_of_study']
                    if isinstance(cat_list, list):
                        for cat in cat_list:
                            cat_dict[cat] += 1
                    # citation
                    citation_dict[len(temp_data['inbound_citations'])] += 1
                    ref_dict[len(temp_data['outbound_citations'])] += 1

                    # journal
                    journal_dict[temp_data['journal']] += 1
                    # venue
                    venue_dict[temp_data['venue']] += 1

                    # good_paper
                    if (len(temp_data['outbound_citations']) >= 5) & (len(temp_data['inbound_citations']) >= 0)\
                            & useful_abstract & useful_title & isinstance(cat_list, list) \
                            & (len(temp_data['authors']) > 0):
                        good_paper_dict[1] += 1
                    else:
                        good_paper_dict[0] += 1

    print('count:', all_count)
    print('pdf:', pdf_dict)
    print('title:', title_dict)
    print('abstract:', abstract_dict)
    print('good_paper:', good_paper_dict)
    print('authors:')
    get_count_detail(authors_dict, all_count, data_path + 'stats_authors_count_{}_{}.csv'.format(time_range[0], time_range[1]))
    print('cat:')
    get_count_detail(cat_dict, all_count, data_path + 'stats_cat_count_{}_{}.csv'.format(time_range[0], time_range[1]), True)
    print('citation:')
    get_count_detail(citation_dict, all_count, data_path + 'stats_citation_count_{}_{}.csv'.format(time_range[0], time_range[1]))
    print('ref:')
    get_count_detail(ref_dict, all_count, data_path + 'stats_ref_count_{}_{}.csv'.format(time_range[0], time_range[1]))
    print('journal:')
    get_count_detail(journal_dict, all_count, data_path + 'stats_journal_count_{}_{}.csv'.format(time_range[0], time_range[1]), True)
    print('venue:')
    get_count_detail(venue_dict, all_count, data_path + 'stats_venue_count_{}_{}.csv'.format(time_range[0], time_range[1]), True)


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
        # get_info_dict('./data/')
        # cite_year_count('./data/')
        # get_split_info_dict('./data/')
        # show_data('./data')
        # show_abstract_stats('./data/')
        show_selected_stats('./data/', [1981, 2021])
    elif args.phase == 'reorganize_data':
        reorganize_data(args.data_path)
        print('reorganize data done.')
    elif args.phase == 'get_info_dict':
        get_info_dict(args.data_path)
        print('get_info_dict done.')
    elif args.phase == 'get_cite_dict':
        get_cite_dict(args.data_path)
        print('get_cite_dict done.')
    elif args.phase == 'cite_year_count':
        cite_year_count(args.data_path)
        print('cite_year_count done.')
    elif args.phase == 'get_split_info_dict':
        get_split_info_dict(args.data_path)
        print('get_split_info_dict done.')
    elif args.phase == 'show_data':
        show_data(args.data_path)
        print('show_data done.')
    elif args.phase == 'show_abstract_stats':
        show_abstract_stats(args.data_path)
        print('show_abstract_stats done.')
    elif args.phase == 'show_selected_stats':
        show_selected_stats(args.data_path, [1980, 2021])
        print('show_abstract_stats done.')
