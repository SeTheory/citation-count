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


def cite_year_count(data_path):
    with open(data_path + 'all_cite.lines') as fr:
        cite_data = list(map(lambda x: eval(x), fr.readlines()))
    with open(data_path + 'all_info.lines') as fr:
        info_data = list(map(lambda x: eval(x), fr.readlines()))
    cite_dict = {x[0]: x[1] for x in cite_data}
    json.dump(cite_dict, open(data_path + 'all_cite_dict.json', 'w+'))
    del cite_data
    info_dict = {}
    for i in range(len(info_data)):
        cur_data = info_data.pop()
        info_dict[cur_data['paper_id']] = cur_data
    json.dump(info_dict, open(data_path + 'all_info_dict.json', 'w+'))
    year_dict = {}
    for paper in cite_dict:
        cur_valid = [paper for paper in cite_dict[paper] if paper in info_dict]
        year_dict[paper] = list(filter(lambda x: x, map(lambda x: info_dict[x]['year'], cur_valid)))
    json.dump(year_dict, open(data_path + 'all_cite_year.json', 'w+'))



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
        cite_year_count('./data/')
    if args.phase == 'reorganize_data':
        reorganize_data(args.data_path)
        print('reorganize data done.')
    if args.phase == 'cite_year_count':
        cite_year_count(args.data_path)
        print('cite_year_count done.')
