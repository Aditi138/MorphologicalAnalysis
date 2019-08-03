from __future__ import division
import argparse
from util import * 
from collections import Counter
import codecs
import mapTaskData
import util
import os
import sys

def get_statistics(tag_lines):

    wordCount = 0 
    pos2count = Counter()
    attribute2count = Counter()
    pos2attribute2count = defaultdict(Counter) 
    attribute2value2count = defaultdict(Counter)
    posPlusAttribute2value2count = defaultdict(Counter)
    
    for tags in tag_lines:
        for tag in tags:

            # Get POS and check if annotation is blank
            current_pos = [t.split('=')[1] for t in tag.strip().split('|') if t.split('=')[0]=="POS"]
            if len(current_pos) == 0:
                continue

            # Update word and POS counts
            wordCount += 1            
            current_pos = current_pos[0]
            pos2count[current_pos] += 1
            
            # Update (attribute, value) counts
            attributeValueTuples = [t.split('=') for t in tag.strip().split('|') if t.split('=')[0]!="_"]
            for attribute, value in attributeValueTuples:
                if value == "GPAUC":
                    print(tags)
                    print(language)
                attribute2value2count[attribute][value] += 1
                if attribute != 'POS':
                    posPlusAttribute2value2count[(current_pos, attribute)][value] += 1

            # Update attribute counts
            other_attributes = [t[0] for t in attributeValueTuples if t[0]!="POS"] 
            attribute2count += Counter(other_attributes)
            pos2attribute2count[current_pos] += Counter(other_attributes)


    return wordCount, pos2count, attribute2count, pos2attribute2count, attribute2value2count, posPlusAttribute2value2count

def get_global_features(ordered_attributes, attribute2count, wordCount):

    feats = []
    for attribute in ordered_attributes:
        feats.append(attribute2count[attribute]/wordCount)
    
    return feats

def get_poswise_features(ordered_attributes, ordered_postags, pos2attribute2count, pos2count):

    feats = []
    for postag in ordered_postags:
        for attribute in ordered_attributes:
            if attribute != 'POS':
                if pos2count[postag] == 0:
                    feats.append(0)
                else:
                    feats.append(pos2attribute2count[postag][attribute]/pos2count[postag])

    return feats

def get_attribute_ordinality_features(attribute_list, attribute2value2count):

    feats = []
    for attribute in attribute_list:
        feats.append(len(attribute2value2count[attribute].keys()))
    return feats

def get_pos_and_value_features(attributes, posPlusAttribute2value2count, pos2count, attribute):

    feats = []
    for val1 in sorted(list(attributes['POS'])):
        for val2 in sorted(list(attributes[attribute])):
            if pos2count[val1] == 0:
                feats.append(0)
            else:
                feats.append(posPlusAttribute2value2count[(val1, attribute)][val2]/pos2count[val1])
            
    return feats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infiles')
    parser.add_argument('--languages')
    parser.add_argument('--outfile')
    parser.add_argument('--attributes', default="../data/attributes.txt")
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--poswise_features', action='store_true')
    parser.add_argument('--vector_type')
    args = parser.parse_args()

    # Load attributes
    attributes = parseAttributes(args.attributes)
    ordered_attributes = sorted(attributes.keys())
    ordered_postags = sorted(list(attributes['POS']))

    print attributes.keys()

    # Make list of features for the vector
    if args.vector_type == "attribute_wise":
        feat_names = ordered_attributes + ['GenderOrdinality', 'NumberOrdinality', 'PersonOrdinality']
        if args.poswise_features:
            for p in ordered_postags:
                for a in ordered_attributes:
                    feat_names.append(p+'_'+a)

    elif args.vector_type == "value_wise":
        feat_names = []
        focus_features = ['Gender', 'Number', 'Case', 'Person']
        for feat in focus_features:    
            for postag in ordered_postags:
                feat_names += [postag+"_"+feat+"_"+val for val in sorted(list(attributes[feat]))]

    else:
        print "Unknown vector type"
        sys.exit()

    # Make vectors
    infiles, languages = args.infiles.split(','), args.languages.split(',')
    assert  len(infiles) == len(languages)
    language2vector = {}

    for language, infile in zip(languages, infiles):
        _, tag_lines = parseTrain(infile)
        wordCount, pos2count, attribute2count, pos2attribute2count, attribute2value2count, posPlusAttribute2value2count = get_statistics(tag_lines)
        
        if args.vector_type == "attribute_wise":
            language2vector[language] = get_global_features(ordered_attributes, attribute2count, wordCount)
            language2vector[language] += get_attribute_ordinality_features(['Gender', 'Number', 'Person'], attribute2value2count)
            if args.poswise_features:
                language2vector[language] += get_poswise_features(ordered_attributes, ordered_postags, pos2attribute2count, pos2count)
        
        else:
            language2vector[language] = []
            for feat in focus_features:
                language2vector[language] += get_pos_and_value_features(attributes, posPlusAttribute2value2count, pos2count, feat)

    # Prune vectors if required
    indices_to_include = set()
    for language, vector in language2vector.iteritems():
        if args.prune:
            non_zero_indices = [i for i, v in enumerate(vector) if v > 0]
            indices_to_include.update(non_zero_indices)
        else:
            indices_to_include.update(range(len(vector)))
            break
    print "Writing vectors of length", len(indices_to_include)

    
    # Write to outfile
    with open(args.outfile, 'w') as outfile:
        outfile.write("Descriptors"+' '+' '.join([f for i, f in enumerate(feat_names) if i in indices_to_include])+'\n')
        for language, vector in language2vector.iteritems():
            outfile.write(language+' '+' '.join([str(v) for i, v in enumerate(vector) if i in indices_to_include])+'\n')
