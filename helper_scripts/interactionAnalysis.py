from __future__ import division
import argparse
from util import * 
from collections import Counter
import codecs
import mapTaskData
import util
import os

def print_correlations(tag_lines, ordered_attribute_counter=None):

    attribute_counter = Counter()
    attribute_pair_counter = {}

    for tags in tag_lines:
        for tag in tags:
            attributes = [t.split('=')[0] for t in tag.strip().split('|')]
            attributes = [a for a in attributes if a != '_']
            for a1 in attributes:
                attribute_counter[a1] += 1
                if a1 not in attribute_pair_counter:
                    attribute_pair_counter[a1] = Counter()        
                for a2 in attributes:
                    attribute_pair_counter[a1][a2] += 1   
                    
    if ordered_attribute_counter is None:
        ordered_attribute_counter = attribute_counter

    print " ", " ".join([attribute for attribute in ordered_attribute_counter.keys()])
    for a1 in ordered_attribute_counter.keys():
        total_count = attribute_counter[a1]
        print a1,
        for a2 in ordered_attribute_counter.keys():
            print round(attribute_pair_counter[a1][a2]/total_count,3),
        print
    print

    return attribute_counter

def get_counts(tag_lines, attributes, verbose=False):

    pos2count = Counter()
    pos2attribute2count = {} 
    attribute2pos2count = {attribute: Counter() for attribute in attributes if attribute != "POS"}
    
    for tags in tag_lines:
        for tag in tags:
            current_pos = [t.split('=')[1] for t in tag.strip().split('|') if t.split('=')[0]=="POS"]
            if len(current_pos) == 0: # Mostly, blank annotations
                continue
            
            current_pos = current_pos[0]
            pos2count[current_pos] += 1
            
            other_attributes = [t.split('=')[0] for t in tag.strip().split('|') if t.split('=')[0]!="POS" and t.split('=')[0]!="_"]
            for attribute in other_attributes: 
                attribute2pos2count[attribute][current_pos] += 1
            if current_pos not in pos2attribute2count:
                pos2attribute2count[current_pos] = Counter()
            pos2attribute2count[current_pos] += Counter(other_attributes)

    if verbose:
        for key, ap2count in attribute2pos2count.iteritems():
            if len(ap2count) > 0:
                print key
                for val, count in ap2count.iteritems():
                    print val+':'+str(count/pos2count[val]) 
                print

    return {'pos2count': pos2count, 
            'pos2attribute2count': pos2attribute2count, 
            'attribute2pos2count': attribute2pos2count
            }

def check_pos_constraints(tag_lines, attribute2pos2count):

    violated_constraints = Counter()

    for tags in tag_lines:
        for tag in tags:
            current_pos = [t.split('=')[1] for t in tag.strip().split('|') if t.split('=')[0]=="POS"]
            if len(current_pos) == 0:
                continue
            
            current_pos = current_pos[0]
            other_attributes = [t.split('=')[0] for t in tag.strip().split('|') if t.split('=')[0]!="POS" and t.split('=')[0]!="_"]            
            for attribute in other_attributes:
                if current_pos not in attribute2pos2count[attribute]:
                    violated_constraints[(current_pos, attribute)] += 1

    return violated_constraints

def postprocess_apply_constraints(input, output, inverseAttributes, constraints, inverse_constraints=False):
    with codecs.open(input, "r", encoding='utf-8') as fin, codecs.open(output, "w", encoding='utf-8') as fout:
        first = True
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue

            if line == "" or line == "\n":
                fout.write("\n")

            else:
                info = line.strip().split("\t")
                feats = info[5].split(";")

                if info[5] == "_":
                    new_feats = "_"
                else:
                    # find POS
                    pos = None
                    for feat in feats:
                        if feat in inverseAttributes and inverseAttributes[feat] == 'POS':
                            pos = feat
                            break

                    # discard features that don't go with POS
                    new_feats = []
                    for feat in feats:
                        if feat in inverseAttributes:
                            if inverse_constraints:
                                if not ( inverseAttributes[feat] in constraints and pos in constraints[inverseAttributes[feat]] ):
                                    new_feats.append(feat)
                            else:
                                if not ( inverseAttributes[feat] in constraints and pos not in constraints[inverseAttributes[feat]] ):
                                    new_feats.append(feat)
                        else:
                            new_feats.append(feat)

                    if new_feats == 0:
                        print("ERROR, new_feats == 0")
                        exit(-1)
                    new_feats = ";".join(new_feats)

                info[5] = new_feats
                fout.write("\t".join(info) + "\n")

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument('--train', default="../data/UD_Belarusian/udmap_be_hse-um-train.conllu")
parser.add_argument('--gold', default="../data/UD_Belarusian/udmap_be_hse-um-dev.conllu")
parser.add_argument('--preds', default="../../analysis/bl_independent_output.conll")
parser.add_argument('--attributes', default="../data/attributes.txt")
parser.add_argument('--output', default="../../analysis/bl_automatic_constrained.conll")
parser.add_argument('--threshold', type=float)
args = parser.parse_args()

# Load attributes
attributes = parseAttributes(args.attributes)
inverseAttributes = parseInverseAttributes(args.attributes)

# Get predictions in right format
base_path, input_file  = args.preds.rsplit('/', 1)
output_file = base_path + "/" + "udmap_" + input_file
mapTaskData.convert(args.preds, output_file, inverseAttributes)

# Read all the data
_, train_tag_lines = parseTrain(args.train)
_, gold_tag_lines = parseTrain(args.gold)
_, pred_tag_lines = parseTrain(output_file)

# gold_attribute_counter = print_correlations(gold_tag_lines)
# print_correlations(pred_tag_lines, gold_attribute_counter)

# check_hard_constraints(train_tag_lines + gold_tag_lines, attributes)
# check_hard_constraints(pred_tag_lines, attributes)

# Just check stuff in the train file

count_stuff = {}

count_stuff['train'] = get_counts(train_tag_lines, attributes)
count_stuff['gold'] = get_counts(gold_tag_lines, attributes)
count_stuff['pred'] = get_counts(pred_tag_lines, attributes)

to_remove_constraints = defaultdict(set)
for attribute in count_stuff['train']['attribute2pos2count'].keys():
    for pos in count_stuff['train']['pos2count'].keys():    
        train_fraction = count_stuff['train']['attribute2pos2count'][attribute][pos]/count_stuff['train']['pos2count'][pos]
        if train_fraction < args.threshold:
            try: # only fails if div by zero, doesn't matter then
                pred_fraction = count_stuff['pred']['attribute2pos2count'][attribute][pos]/count_stuff['pred']['pos2count'][pos]
                if pred_fraction > 0:
                    print attribute, pos, train_fraction, pred_fraction
                    to_remove_constraints[attribute].add(pos)
            except:
                pass
# postprocess_apply_constraints(args.preds, args.output, inverseAttributes, to_remove_constraints, inverse_constraints=True)

# focus_attributes = ['Aspect', 'Finiteness', 'Voice', 'Tense', 'VerbForm']
# for attribute in focus_attributes:
#     print attribute
#     for key, stuff in count_stuff.iteritems():
#         print key
#         for val, count in stuff[2][attribute].iteritems():
#             print val, ':', str(count/stuff[0][val]) 
#         print

# constraints = {'Aspect':set('V'), 'Finiteness':set('V'), 'Voice':set('V'), 'Tense':set('V'), 'VerbForm':set('V')}
# postprocess_apply_constraints(args.preds, args.output, inverseAttributes, constraints)
