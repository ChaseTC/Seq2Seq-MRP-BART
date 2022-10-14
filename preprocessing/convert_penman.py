from pathlib import Path
import json
import logging

import penman
from delphin.util import _bfs

logger = logging.getLogger(__name__)
in_path = Path('data\extracted')
out_path = Path('data\penman')
splits = ['dev', 'test', 'train']

def create_nodes_ids(nodes):
    node_ids = []
    for node in nodes:
        anchors = node['anchors'][0]
        anchor_string = '< ' + str(anchors['from']) + ' ' + str(anchors['end']) + ' >'
        node_ids.append(anchor_string + ' ' + node['label'])
    return node_ids

def to_penman(input_dict):
    nodes = input_dict['nodes']
    edges = input_dict['edges']

    # determine if graph is connected
    g = {i: set() for i in range(len(nodes))}
    for edge in edges:
        source = edge['source']
        target = edge['target']
        g[source].add(target)
        g[target].add(source)
    main_component = _bfs(g, start=input_dict['tops'][0])
    complete = True

    triples = []
    node_ids = create_nodes_ids(input_dict['nodes'])
    
    for node in nodes:
        triple = (node_ids[node['id']], ':instance', None)
        if node['id'] in main_component:
            triples.append(triple)
        else:
            complete = False


    for edge in edges:
        source = edge['source']
        target = edge['target']
        triple = (node_ids[source], ':' + edge['label'], node_ids[target])
        if source in main_component and target in main_component:
            triples.append(triple)
        else:
            complete = False

    if not complete:
        logger.warning('disconnected graph cannot be completely encoded: %s', input_dict['input'])

    top = node_ids[input_dict['tops'][0]]
    
    g = penman.Graph(triples, top)
    return penman.encode(g, indent=None)

def process_line(line):
    input_dict = json.loads(line)
    j_dict = {'sentence': input_dict['input'],
              'penman': to_penman(input_dict)}
    return json.dumps(j_dict)

def main():
    if not out_path.exists():
        out_path.mkdir()
    for split in splits:
        if not (out_path / split).exists():
            (out_path / split).mkdir()

    for split in splits:
        for file in (in_path / split).iterdir():
            j_lines = []
            with open(file, 'r') as f:
                for line in f.readlines():
                    j_lines.append(process_line(line))

            filename = file.name.split('\\')[-1].split('.')[0] + '.json'
            with open(out_path/split/filename, 'w') as f:
                for line in j_lines:
                    if line != "":
                        f.write(line + "\n")

if __name__ == '__main__':
    main()
