import copy

DATASET_DIRPATH='./dataset/'
CDR_FILE_PREFIX='CDR_'
CDR_FILE_SUFFIX='Set.PubTator.txt'

#corpus_pubtator_pmids_trng.txt
from_bc5cdrset_2_common = {
    "Training": "trng",
    "Development": "dev",
    "Test": "test"
}
PMID_FILE_PREFIX='corpus_pubtator_pmids_'
PMID_FILE_SUFFIX='.txt'

def trn_dev_test_pmidsets_maker():
    all_pmids = []
    for bc5cdr_file, pmid_file_symbol in from_bc5cdrset_2_common.items():
        source_file = DATASET_DIRPATH + CDR_FILE_PREFIX + bc5cdr_file + CDR_FILE_SUFFIX
        dest_file = DATASET_DIRPATH + PMID_FILE_PREFIX + pmid_file_symbol + PMID_FILE_SUFFIX

        pmids = []
        with open(source_file, 'r') as f:
            for line in f:
                if '|t|' in line:
                    l = line.strip().split('|')
                    pmid = l[0]
                    pmids.append(pmid)
                    all_pmids.append(pmid)

        with open(dest_file, 'w') as g:
            for idx, pmid in enumerate(pmids):
                if idx != len(pmids) -1:
                    g.write(pmid+'\n')
                else:
                    g.write(pmid)

    with open(DATASET_DIRPATH + 'corpus_pubtator_pmids_all.txt', 'w') as h:
        for idx, pmid in enumerate(all_pmids):
            if idx != len(all_pmids) - 1:
                h.write(pmid + '\n')
            else:
                h.write(pmid)


def corpus_pubtator_maker():
    entire_file = ''
    one_title_and_abst = ''
    dest_file = DATASET_DIRPATH + 'corpus_pubtator.txt'

    for bc5cdr_file, pmid_file_symbol in from_bc5cdrset_2_common.items():
        source_file = DATASET_DIRPATH + CDR_FILE_PREFIX + bc5cdr_file + CDR_FILE_SUFFIX

        with open(source_file, 'r') as f:
            for line in f:
                if line.strip() == "":
                    entire_file += one_title_and_abst + '\n'
                    one_title_and_abst = copy.copy('')
                else:
                    if '|t|' in line:
                        one_title_and_abst += line
                    elif '|a|' in line:
                        one_title_and_abst += line
                    else:
                        if 'CID' in line:
                            continue
                        if line.strip().split('\t')[5] == '-1':
                            continue
                        if len(line.strip().split('\t')[5].split('|')) != 1:
                            continue

                        one_title_and_abst += line

    with open(dest_file, 'w') as g:
        g.write(entire_file)


if __name__ == '__main__':
    trn_dev_test_pmidsets_maker()
    corpus_pubtator_maker()