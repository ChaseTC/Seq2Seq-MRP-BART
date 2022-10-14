import penman
import re
import json
from parser import Parser
from pathlib import Path
from collections import defaultdict
from delphin.util import _bfs
import smatch
import argparse

start_edm = False
use_smatch = True

def balance_brackets(string):
    string = string.rstrip('\n')
    l_count = 0
    r_count = 0
    for c in string:
        if c == '(':
            l_count += 1
        elif c == ')':
            r_count += 1
    if r_count < l_count:
        return string + ')'*(l_count - r_count)
    else:
        return string

def preprocess(penman):
    penman = penman.replace('< ', '<').replace(' > ', '>').replace(' )', ')')
    return re.sub(r"\s(?=\d+>)", '-', penman)

def node_from_string(src, id):
    label = src.split('>')[1]
    source = int(src.split('>')[0].split('-')[0].lstrip('<'))
    target = int(src.split('>')[0].split('-')[-1].lstrip('<'))
    return {'id': id, 'anchors': [{'from': source, 'end': target}], 'label': label}

def create_nodes_ids(nodes):
    node_ids = []
    for node in nodes:
        anchors = node['anchors'][0]
        anchor_string = '< ' + str(anchors['from']) + ' ' + str(anchors['end']) + ' >'
        node_ids.append(anchor_string + ' ' + node['label'])
    return node_ids

def to_amr(input_dict):
    nodes = input_dict['nodes']
    edges = input_dict['edges']

    g = {i: set() for i in range(len(nodes))}
    for edge in edges:
        source = edge['source']
        target = edge['target']
        g[source].add(target)
        g[target].add(source)
    main_component = _bfs(g, start=input_dict['tops'][0])

    triples = []
    node_ids = create_nodes_ids(input_dict['nodes'])
    
    for node in nodes:
        if node['id'] in main_component:
            triple = (node_ids[node['id']], ':instance', node['label'])
            triples.append(triple)

    for edge in edges:
        source = edge['source']
        target = edge['target']
        triple = (node_ids[source], ':' + edge['label'], node_ids[target])
        if source in main_component and target in main_component:
            triples.append(triple)

    top = node_ids[input_dict['tops'][0]]
    triples.sort()

    g = penman.Graph(triples, top)
    return penman.encode(g)

def from_triples(triples):
    nids = {}
    nodes = []
    edges = []

    curr_id = 0
    for src, rel, tgt in triples:
        if src not in nids.keys():
            nids[src] = curr_id
            nodes.append(node_from_string(src, curr_id))
            curr_id += 1
        if tgt is not None and tgt not in nids.keys():
            nids[tgt] = curr_id
            nodes.append(node_from_string(tgt, curr_id))
            curr_id += 1
        if rel != ':instance':
            edge = {'source': nids[src], 'target': nids[tgt], 'label': rel.lstrip(':')}
            edges.append(edge)
    g_dict = {'nodes': nodes, 'edges': edges, 'tops': [nodes[0]['id']]}
    return g_dict

def process_penman(p_string):
    g = penman.decode(preprocess(p_string))
    return from_triples(g.triples)

def node_from_json(node):
    frm = str(node['anchors'][0]['from'])
    end = str(node['anchors'][0]['end'])
    label = node['label']
    return frm + ' ' + end + ' ' + label

def edge_from_json(nodes, edge):
    src = edge['source']
    tgt = edge['target']
    s_frm = str(nodes[src]['anchors'][0]['from'])
    s_end = str(nodes[src]['anchors'][0]['end'])
    t_frm = str(nodes[tgt]['anchors'][0]['from'])
    t_end = str(nodes[tgt]['anchors'][0]['end'])
    label = edge['label']
    if start_edm:
        return s_frm + ' ' + label + ' ' + t_frm
    else:
        return s_frm + ' ' + s_end + ' ' + label + ' ' + t_frm + ' ' + t_end
    

    
def evaluate(gold, predicted):
    if predicted is None:
        return
    
    gold_nodes = set()
    gold_edges = set()
    pre_nodes = set()
    pre_edges = set()

    for node in gold['nodes']:
        gold_nodes.add(node_from_json(node))
    for edge in gold['edges']:
        gold_edges.add(edge_from_json(gold['nodes'], edge))
    for node in predicted['nodes']:
        pre_nodes.add(node_from_json(node))
    for edge in predicted['edges']:
        pre_edges.add(edge_from_json(predicted['nodes'], edge))

    node_tp, node_fp = 0, 0
    edge_tp, edge_fp = 0, 0

    # get node positives
    while(len(pre_nodes) > 0):
        node = pre_nodes.pop()
        if node in gold_nodes:
            node_tp += 1
            gold_nodes.remove(node)
        else:
            node_fp += 1
    
    # nodes remaining in the set are false negatives
    node_fn = len(gold_nodes)

    # get edge positives
    while(len(pre_edges) > 0):
        edge = pre_edges.pop()
        if edge in gold_edges:
            edge_tp += 1
            gold_edges.remove(edge)
        else:
            edge_fp += 1
    
    # edges remaining in the set are false negatives
    edge_fn = len(gold_edges)

    # node precision
    if node_tp + node_fp == 0:
        node_precision = 1
    else:
        node_precision = node_tp / (node_tp + node_fp)
    
    # node recall
    if (node_tp + node_fn) == 0:
        node_recall = 1
    else:
        node_recall = node_tp / (node_tp + node_fn)

    # node f1
    if (node_precision + node_recall) == 0:
        node_f1 = 0
    else:
        node_f1 = 2 * (node_precision * node_recall)/(node_precision + node_recall)
    
    # edge precision
    if edge_tp + edge_fp == 0:
        edge_precision = 1
    else:
        edge_precision = edge_tp / (edge_tp + edge_fp)

    # edge recall
    if (edge_tp + edge_fn) == 0:
        edge_recall = 1
    else:
        edge_recall = edge_tp / (edge_tp + edge_fn)
    
    # edge f1
    if (edge_precision + edge_recall) == 0:
        edge_f1 = 0    
    else:
        edge_f1 = 2 * (edge_precision * edge_recall)/(edge_precision + edge_recall)
    
    precision = (node_precision + edge_precision) / 2
    recall = (node_recall + edge_recall) / 2
    f1 = (node_f1 + edge_f1) / 2

    return {'precision': precision,
            'recall': recall,
            'f1 score': f1,
            'node precision': node_precision,
            'node recall': node_recall,
            'node f1 score': node_f1,
            'edge precision': edge_precision,
            'edge recall': edge_recall,
            'edge f1 score': edge_f1}


def run(args):
    out_dir = Path('data/eval/')

    if not out_dir.exists():
        out_dir.mkdir()

    gold_graphs = []
    predicted_graphs = []
    failed = []
    for file in Path('data/extracted/test').iterdir():
            with open(file, 'r') as f:
                for line in f.readlines():
                    gold_graphs.append(json.loads(line))

    with open(args.input, 'r') as f:
        for i, line in enumerate(f.readlines()):
            p_string = balance_brackets(line)
            try:
                predicted_graphs.append(process_penman(p_string))
            except:
                parsed = Parser(line).parse()
                try:
                    predicted_graphs.append(process_penman(parsed))
                except:
                    failed.append((i, line))
                    predicted_graphs.append(None)
    if failed:
        with open(out_dir/'failed_gold.txt', 'w') as f:
            for i, _ in failed:
                f.write(json.dumps(gold_graphs[i]) + '\n')
        with open(out_dir/'failed_predicted.txt', 'w') as f:
            for i, p_string in failed:
                f.write(p_string)
    evaluations = []
    totals = defaultdict(float)
    for i in range(len(gold_graphs)):
        e = evaluate(gold_graphs[i], predicted_graphs[i])

        if use_smatch:
            p = iter(to_amr(predicted_graphs[i]).splitlines())
            g = iter(to_amr(gold_graphs[i]).splitlines())
            try:
                smatch_score = next(smatch.score_amr_pairs(p, g))
                e['smatch'] = smatch_score[2]
            except:
                e['smatch'] = 0
        
        for k, v in e.items():
            totals[k] += v
        evaluations.append(json.dumps(e))

    if start_edm:
        outfile = 'start_eval-' + args.input
    else:
        outfile = 'eval-' + args.input

    with open(out_dir/outfile, 'w') as f:
        for e in evaluations:
            f.write(e + '\n')
        for k, v in totals.items():
            totals[k] = v/len(evaluations)
        f.write('\n' + json.dumps(totals))

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('input', help='directory path to file containing predictions')
    argparser.add_argument('--start_edm', action='store_true', help='uses start EDM instead of EDM')
    argparser.add_argument('--no_smatch', action='store_true', help='disables smatch')

    args = argparser.parse_args()
    if args.start_edm:
        global start_edm
        start_edm = True
    if args.no_smatch:
        global use_smatch
        use_smatch = False
    run(args)
    
if __name__ == '__main__':
    main()