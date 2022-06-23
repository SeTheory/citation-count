import argparse
import datetime
import json
import os


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
        get_split_info_dict('./data/')
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
