import codecs
import argparse
from util import parseInverseAttributes
import os
'''
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="/Users/aditichaudhary/Documents/CMU/sigmorph_data/task2/UD_Italian-ParTUT/")
parser.add_argument("--attributes",type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/myNRF/data/attributes.txt")
parser.add_argument("--mode", type=str, default="evaluate", help=["train", "evaluate"])
parser.add_argument("--prediction_output", type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/outputs/marathi_mono_pos_emb_from_english.model_baseline_mr-1600_test_output.conll")
parser.add_argument("--dev", type=str, default="/Users/aditichaudhary/Documents/CMU/sigmorph_data/task2/UD_Marathi-UFAL/mr_ufal-um-dev.conllu")
parser.add_argument("--pos", type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/Analysis/UD_Marathi/unimorph_pos_file_dev.conllu")

args = parser.parse_args()
'''
def convert(input, output, inverseAttributes):
    with codecs.open(input, "r", encoding='utf-8') as fin, codecs.open(output, "w", encoding='utf-8') as fout:
        first = True
        for line in fin:
            if line.startswith("#"):
                #fout.write(line)
                continue

            if line == "" or line == "\n":
                fout.write("\n")

            else:
                info = line.strip().split("\t")
                feats = info[5].split(";")

                if info[5] == "_":
                    new_feats = "_"
                elif info[5] == "B-UNK":
                    new_feats = "B-UNK"
                else:
                    new_feats = []
                    categores_covered = {}
                    for feat in feats:
                        if feat in inverseAttributes:
                            if inverseAttributes[feat] in categores_covered:
                                print("WARNING: multiple values for same category", inverseAttributes[feat], feat,
                                      categores_covered[inverseAttributes[feat]])
                            new_feats.append(inverseAttributes[feat] + "=" + feat)
                            categores_covered[inverseAttributes[feat]] = feat
                        else:
                            if "NewCategory" in categores_covered:
                                print("WARNING: multiple values for same category", "NewCategory", feat,
                                      categores_covered["NewCategory"])
                                new_feats.append("NewCategory" + "=" + feat)
                                categores_covered["NewCategory"] = feat

                    if len(new_feats) == 0:
                        print("ERROR, new_feats == 0")
                        exit(-1)
                    new_feats = "|".join(new_feats)

                info[5] = new_feats
                fout.write("\t".join(info) + "\n")

def convertToOriginal(input, token_col=0, feats_col=1):
    lines, one_line = {}, []
    one_prediction = []
    with codecs.open(input, "r", encoding='utf-8') as fin:
        for line in fin:
            if line == "" or line == "\n":
                assert len(one_line) == len(one_prediction)
                lines[" ".join(one_line)] = " ".join(one_prediction)

                one_line = []
                one_prediction = []
            else:
                info = line.strip().split("\t")
                if len(info) > 1:
                    if info[feats_col] == "":
                        info[feats_col] = "_"
                    token, feats = info[token_col], info[feats_col]
                else:
                    token, feats = info[token_col],"_"
                one_line.append(token)

                if feats == "_":
                    parsed_feats = "_"
                else:
                    parsed_feats = []
                    for feat in feats.split("|"):
                        feat_value = feat.split("=")[-1]
                        if feat_value != "_":
                            parsed_feats.append(feat_value)
                    if len(parsed_feats) >0:
                        parsed_feats = ";".join(parsed_feats)
                    else:
                        parsed_feats = "_"
                one_prediction.append(parsed_feats)

    return lines


def mapFiles(gold_file, output_mapped_file, lines, gold_pos= None, pos_tags = None):
    with codecs.open(gold_file, "r", encoding='utf-8') as fin, \
            codecs.open(output_mapped_file, "w", encoding='utf-8') as fout:
        one_line = []
        line_num =0
        for line in fin:

            if line.startswith("#"):
                continue

            if line == "" or line == "\n":
                sent = " ".join(one_line)

                pred_feats = lines[sent]
                info = ["_" for _ in range(10)]
                i = 1
                # if pos_tags is not None:
                #     one_pos = pos_tags[line_num]

                for token, feats in zip(sent.split(), pred_feats.split()):
                    # Process feats and pos
                    info[0], info[1], info[5] = str(i), token, feats

                    # if pos_tags is not None:
                    #     pos_token = one_pos[i-1]
                    #     info[5] = pos_token + ";" + info[5]

                    fout.write("\t".join(info) + "\n")
                    i += 1
                fout.write("\n")

                one_line = []
                line_num +=1

            else:
                items = line.strip().split("\t")
                one_line.append(items[1])

    if gold_pos is not None:
        with codecs.open(gold_file, "r", encoding='utf-8') as fin, \
                codecs.open(gold_pos, "w", encoding='utf-8') as fout:
            for line in fin:
                if line.startswith("#"):
                    continue
                if line == "" or line == "\n":
                    fout.write("\n")
                else:
                    info = line.strip().split("\t")
                    info[5] = info[5].split("=")[-1]
                    fout.write("\t".join(info) + "\n")

if __name__ == "__main__":

    '''Instructions 
    To map the UniMorph data into key:value type data, set args.mode = train and specify the dir of the language in args.input = ../UD_English/
    
    To map the output from NRF model, set args.mode = evaluate and specify the output file in args.prediction_output, it will create the output 
    in the UniMorph schema with suffix mapped_ordered.conll
    
    The map for conversion is present in data/attributes.txt to be specified in args.attributes
    '''
    if args.mode == "train":
        inverseAttributes = parseInverseAttributes(args.attributes)
        for [path, dir, files] in os.walk(args.input):
            for file in files:
                if file.endswith("train.conllu") and not file.startswith("mapped"):
                    output_file = args.input +"/" + "udmap_" + file
                    convert(args.input + "/" + file, output_file,inverseAttributes)
                if file.endswith("dev.conllu") and not file.startswith("mapped"):
                    output_file = args.input +"/" + "udmap_" + file
                    convert(args.input + "/" + file, output_file,inverseAttributes)

    elif args.mode == "evaluate":
        mapped_prediction = args.prediction_output.split(".conll")[0] + "_mapped_ordered.conll"
        lines = convertToOriginal(args.prediction_output)
        mapFiles(args.dev, mapped_prediction, lines, args.pos)
