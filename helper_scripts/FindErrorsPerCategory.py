import codecs
import argparse
from collections import defaultdict
import glob, os
from util import *


parser = argparse.ArgumentParser()
#inputs
#parser.add_argument("--prediction",type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/Analysis/hindi_mono.model_baseline_fixed_proper_nouns.txt")
parser.add_argument("--prediction", type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/outputs/marathi_mono_rerunning_pos_emb_from_english.model_baseline_mr-1600_test_output_mapped_ordered.conll")
parser.add_argument("--dev",type=str, default="/Users/aditichaudhary/Documents/CMU/sigmorph_data/task2/UD_Marathi-UFAL/mr_ufal-um-dev.conllu")
parser.add_argument("--attributes", type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/myNRF/data/attributes.txt")

#outputs
parser.add_argument("--analysis_folder", type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/Analysis/es/")
parser.add_argument("--fixed_pred", type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/Analysis/es/fixed_")
parser.add_argument("--token_errors", type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/Analysis/es/token_errors.txt")



#Check individual outputs
parser.add_argument("--gold_category",type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/myNRF/data/UD_Hindi/split_attributes/Case_gold_dev.conllu")
parser.add_argument("--pred_category", type=str)

args = parser.parse_args()


def findNumberErrors(args):
    lines, gold_values = parseTrain(args.gold_category)
    _, pred_values = parseTrain(args.pred_category)
    assert len(gold_values) == len(pred_values)
    total = 0
    count =0
    gold = 0
    for gold_sent, pred_sent in zip(gold_values, pred_values):
        assert len(gold_sent) == len(pred_sent)
        for gold_token, pred_token in zip(gold_sent, pred_sent):
            if gold_token != pred_token:
                count +=1
            if gold_token != "_":
                gold +=1
            total +=1
    print("{0}/{1}".format(count, total))
    print(gold)

def getOutputFeats():
    gold_map = "/Users/aditichaudhary/Documents/CMU/sigmorph_data/task2/UD_Hindi-HDTB/pretrain_pos_udmap_hi_hdtb-um-dev.conllu"
    feat_gold = "/Users/aditichaudhary/Documents/CMU/sigmorph_data/task2/UD_Hindi-HDTB/pretrain_pos_um_hi_hdtb-um-dev.conllu"
    with codecs.open(gold_map, "r", encoding='utf-8') as fin, codecs.open(feat_gold, "w", encoding='utf-8') as fout:
        for line in fin:
            if line == "" or line == "\n":
                fout.write("\n")
            else:
                line = line.strip().split("\t")
                feats = line[5]
                new_feats = []
                if feats == "_":
                    new_feats = "_"
                else:
                    feats = feats.split("|")
                    for feat in feats:
                        value = feat.split("=")[-1]
                        new_feats.append(value)
                    new_feats = ";".join(new_feats)
                line[5] = new_feats
                fout.write("\t".join(line) + "\n")

def fixErrorsPerCategory(lines, predictions, golds, attributes, category ):
    #print("Fixing errors for attribute: ", category)
    attribute_tagset = attributes[category]
    category_token_error_count = 0
    ref_file_name = args.fixed_pred + category + ".conllu"
    fout = codecs.open(ref_file_name, "w", encoding='utf-8')
    token_pred_count = 0
    for line, pred, gold in zip(lines, predictions, golds):
        for i in range(len(line)):
            token, pred_token, gold_token = line[i], pred[i], gold[i]
            gold_feats = gold_token.split(";")
            gold_value = None

            for gold_feat in gold_feats:
                if gold_feat in attribute_tagset:
                    gold_value = gold_feat
                    break

            pred_feats = pred_token.split(";")
            fixed_feats = []
            pred_value = None


            if pred_feats == "_":
                token_pred_count +=1
                if gold_value:
                    fixed_feats = gold_value
                else:
                    fixed_feats = "_"
            else:
                for pred_feat in pred_feats:

                    if pred_feat in attribute_tagset:
                        if gold_value:#Gold value had that property but pred also predicted it
                            fixed_feats.append(gold_value)
                        pred_value = pred_feat
                        token_pred_count +=1
                        break

                    else:
                        fixed_feats.append(pred_feat)

                if gold_value and not pred_value:
                    fixed_feats.append(gold_value)

            #Count Errors per category
            if gold_value and not pred_value:
                category_token_error_count += 1
            elif pred_value and not gold_value:
                category_token_error_count += 1
            elif pred_value != gold_value:
                category_token_error_count += 1

            info= ["_" for _ in range(10)]
            info[0], info[1], info[5] = str(i + 1), token, ";".join(fixed_feats)
            fout.write("\t".join(info)  + "\n")

        fout.write("\n")

    print("{0} token errors: {1} total: {2}".format(category,category_token_error_count, token_pred_count))
    #_,_, acc, f1, prec, recall = main(ref_file_name, args.dev)
    #print("After fixing :" + category + " the scores are Acc: " + str(acc) + " F1: " + str(f1)+ " Prec: " + str(prec) + " Recall: " + str(recall) )



if __name__ == "__main__":

    dev_lines, dev_gold = parseTrain(args.dev)
    lines,predictins = parseTrain(args.prediction)

    assert len(lines) == len(dev_lines)
    assert len(predictins) == len(dev_gold)


    '''Fix erros by attributes'''
    attributes = parseAttributes(args.attributes)
    for key in attributes.keys():
        #if key == "Polarity":
        fixErrorsPerCategory(lines, predictins, dev_gold, attributes, key)


    #findNumberErrors(args)

    #getOutputFeats()
