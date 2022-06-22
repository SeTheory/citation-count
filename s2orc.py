import argparse
import datetime
import json
import os


def reorganize_data(data_path):
    files = os.listdir(data_path)
    info_fw = open(data_path + 'all_info.lines', 'w+')
    ab_fw = open(data_path + 'all_abstract.lines', 'w+')
    citation_fw = open(data_path + 'all_cite.lines', 'w+')
    reference_fw = open(data_path + 'all_ref.lines', 'w+')
    for file in files:
        with open(data_path + file) as fr:
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
    if args.phase == 'reorganize_data':
        reorganize_data(args.data_path)
        print('reorganize data done.')
