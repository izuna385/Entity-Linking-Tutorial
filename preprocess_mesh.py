import json
import pdb
import os
MESH_PATH = './mesh_2020.jsonl'
MESH_DIRPATH = './mesh/'

def mesh_loader():
    concepts = list()
    with open(MESH_PATH, 'r') as f:
        for line in f:
            dui_one_concept = json.loads(line.strip())
            # dict_keys(['concept_id', 'aliases', 'canonical_name', 'definition'])
            concepts.append(dui_one_concept)
    duis = [concept['concept_id'] for concept in concepts]

    dui2idx, idx2dui = {}, {}
    for idx, dui in enumerate(duis):
        dui2idx.update({dui: idx})
        idx2dui.update({idx: dui})

    dui2canonical, dui2definition = {}, {}
    for concept in concepts:
        dui2canonical.update({concept['concept_id']: concept['canonical_name']})
        if 'definition' in concept:
            dui2definition.update({concept['concept_id']: concept['definition']})
        else:
            dui2definition.update({concept['concept_id']: concept['canonical_name']})

    return dui2idx, idx2dui, dui2canonical, dui2definition

def kb_dumper():
    dui2idx, idx2dui, dui2canonical, dui2definition = mesh_loader()
    dui2idx_path = MESH_DIRPATH + 'dui2idx.json'
    idx2dui_path = MESH_DIRPATH + 'idx2dui.json'
    dui2canonical_path = MESH_DIRPATH + 'dui2canonical.json'
    dui2definition_path = MESH_DIRPATH + 'dui2definition.json'

    with open(dui2idx_path, 'w') as dui2idx_f:
        json.dump(dui2idx, dui2idx_f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

    with open(idx2dui_path, 'w') as idx2dui_f:
        json.dump(idx2dui, idx2dui_f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

    with open(dui2canonical_path, 'w') as dui2canonical_f:
        json.dump(dui2canonical, dui2canonical_f, ensure_ascii=False, indent=4, sort_keys=False, separators=(',', ': '))

    with open(dui2definition_path, 'w') as dui2definition_f:
        json.dump(dui2definition, dui2definition_f, ensure_ascii=False, indent=4, sort_keys=False,
                  separators=(',', ': '))


if __name__ == '__main__':
    if not os.path.exists(MESH_DIRPATH):
        os.mkdir(MESH_DIRPATH)
    kb_dumper()