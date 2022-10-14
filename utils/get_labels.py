from pathlib import Path
from collections import Counter
import json

path = Path('data\extracted')
splits = ['dev','train']

node_labels = []
edge_labels = set()

for split in splits:
    for file in (path / split).iterdir():
        with open(file, 'r') as f:
            for line in f.readlines():
                ex_dict = json.loads(line)
                nodes = ex_dict['nodes']
                edges = ex_dict['edges']
                for node in nodes:
                    node_labels.append(node['label'])
                for edge in edges:
                    edge_labels.add(edge['label'])

with open('edge_labels.txt', 'w') as f:
    for label in edge_labels:
        f.write('":' + label + '", ')
        f.write('":' + label + '-of", ')

node_label_counts = Counter(node_labels)

with open('node_labels.txt', 'w') as f:
    for l,v in node_label_counts.items():
        if v >= 100:
            f.write('"' + l + '", ')
