import codecs
from collections import defaultdict


SEPARATOR = ";"
CATEGORIES = set()
def set_equal(str1, str2):
    set1 = set(str1.split(SEPARATOR))
    set2 = set(str2.split(SEPARATOR))
    return set1 == set2

def parseInput(input):
    lines = []
    predictions = []
    outputs = []
    one_line = []
    one_pred = []
    one_output = []

    with codecs.open(input, "r", encoding='utf-8') as fin:
        for line in fin:
            if line == "" or line == '\n':
                lines.append(one_line)
                predictions.append(one_pred)
                outputs.append(one_output)
                one_line = []
                one_pred = []
                one_output = []
            else:
                info = line.strip().split("\t")
                one_line.append(info[1])
                one_pred.append(info[2])
                one_output.append(info[3])

        return lines, predictions, outputs

def parseMappedInput(input):
    lines = []
    predictions = []
    outputs = []
    one_line = []
    one_pred = []
    one_output = []

    with codecs.open(input, "r", encoding='utf-8') as fin:
        for line in fin:
            if line == "" or line == '\n':
                lines.append(one_line)
                predictions.append(one_pred)
                outputs.append(one_output)
                one_line = []
                one_pred = []
                one_output = []
            else:
                info = line.strip().split("\t")
                one_line.append(info[0])
                pred_feats, gold_feats = info[1].split("|"), info[2].split("|")
                pred_dict, gold_dict = {}, {}
                for feat in pred_feats:

                    key, value = feat.split("=")
                    CATEGORIES.add(key)
                    pred_dict[key] = value
                for feat in gold_feats:

                    key, value = feat.split("=")
                    CATEGORIES.add(key)
                    gold_dict[key] = value

                one_pred.append(pred_dict)
                one_output.append(gold_dict)

        return lines, predictions, outputs

def parseTrain(input):
    with codecs.open(input, "r", encoding='utf-8') as fin:
        lines = []
        outputs = []
        one_line = []
        one_output = []
        tokens = 0
        for line in fin:
            if line.startswith("#"):
                continue
            if line == "" or line == '\n' or line == " ":
                lines.append(one_line)
                outputs.append(one_output)
                one_line = []
                one_output = []
            else:
                info = line.strip().split("\t")
                one_line.append(info[1])
                one_output.append(info[5])
                tokens +=1

        if len(one_line)> 0:
            lines.append(one_line)
            outputs.append(one_output)

        #print(tokens)
        return lines, outputs

def mapFiles(original_dev_file, prediction_file,  mappped_pred_file):

    lines = {}
    with codecs.open(prediction_file, "r", encoding='utf-8') as fin:
        one_line = []
        one_pos = []
        one_feats = []
        for line in fin:
            if line.startswith("#"):
                continue
            if line == "" or line == "\n":
                sent = " ".join(one_line)
                pos = " ".join(one_pos)
                feats = " ".join(one_feats)

                one_line = []
                one_pos = []
                one_feats = []
                if sent not in lines:
                    lines[sent] = (pos, feats)
                else:
                    print("Duplicate sentence found!", sent)

            else:
                info = line.strip().split("\t")

                one_line.append(info[1])
                one_pos.append(info[3])
                one_feats.append(info[5])

    with codecs.open(original_dev_file, "r", encoding='utf-8') as fin,\
            codecs.open(mappped_pred_file, "w", encoding='utf-8') as fout:

        one_line = []
        one_pos = []
        one_feats = []
        for line in fin:

            if line.startswith("#"):
                continue
            if line == "" or line =="\n":
                sent = " ".join(one_line)
                gold_pos = " ".join(one_pos)
                gold_feats = " ".join(one_feats)

                # if sent not in lines:
                #     continue
                (pred_pos, pred_feats) = lines[sent]
                info  = ["_" for _ in range(10)]
                i=1
                for token, pos, feats in zip(sent.split(), pred_pos.split(), pred_feats.split() ):
                    info[0], info[1], info[5] = str(i), token,  feats
                    fout.write("\t".join(info) + "\n")
                    i +=1
                fout.write("\n")


                one_line = []
                one_pos = []
                one_feats = []
            else:
                items = line.strip().split("\t")
                one_line.append(items[1])
                one_pos.append(items[3])
                one_feats.append(items[5])

def findEntities(lines, gold_l, output_file):
    Entities = set()
    for line, gold in zip(lines, gold_l):

        entity = []
        for i in range(len(line)):
            token, gold_token = line[i], gold[i]
            if "PROPN" in gold_token:
                entity.append(token)

            else:
                if len(entity) > 0:
                    Entities.add(" ".join(entity))
                entity = []
    print("Entities: {0}".format(len(Entities)))
    with codecs.open(output_file, "r", encoding='utf-8') as fout:
        for ent in Entities:
            fout.write(ent + "\n")
    return Entities

def visualization(input, output):
    with codecs.open(input, "r", encoding='utf-8') as fin, codecs.open(output, "w", encoding='utf-8') as fout:
        for line in fin:
            if line.startswith("#"):
                continue
            if line == "" or line == "\n":
                fout.write("\n")
            else:
                info = line.strip().split("\t")
                feats = info[5].split(";")
                feats.sort()
                fout.write(info[0] + "\t" + info[1] + "\t" + ";".join(feats) + "\n")

def checkCategory(category,line_num,  catgory_errors_count, category_errors_line, pred, gold):
    if category in gold and category not in pred:
        catgory_errors_count[category] +=1
        category_errors_line[category].add(line_num)

    elif category in gold and category in pred:
        if gold[category] != pred[category]:
            catgory_errors_count[category] +=1
            category_errors_line[category].add(line_num)

def parseAttributes(input):
    attributes = defaultdict(set)
    with codecs.open(input) as fin:
        for line in fin:
            line = line.strip().split()
            attributes[line[0]].add(line[-1].upper())
    return attributes

def parseInverseAttributes(input):
    #{v:POS, n:POS ...}
    inverseAttributes = {}

    with codecs.open(input) as fin:
        for line in fin:
            line = line.strip().split()
            tag_value = line[-1].upper()
            if tag_value in inverseAttributes and inverseAttributes[tag_value] != line[0]:
                print("Same tag value has multiple categores", line)
            else:
                inverseAttributes[tag_value]=line[0]

    return inverseAttributes