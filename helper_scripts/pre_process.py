import codecs, argparse, os
from util import parseInverseAttributes, parseAttributes
import mapTaskData

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=None)
parser.add_argument("--attributes", default="attributes.txt")
parser.add_argument("--train", default="sigmorph_data/task2/UD_Belarusian-HSE/be_hse-um-train.conllu")
parser.add_argument("--dev", default="sigmorph_data/task2/UD_Belarusian-HSE/be_hse-um-dev.conllu")
parser.add_argument("--test", default="/sigmorph_data/task2/UD_Belarusian-HSE/be_hse-um-test.conllu")
parser.add_argument("--max_lines", type=int, default=1000000)
args = parser.parse_args()

print(args)

def convert(input, output, write=True):
    pos_tags = []
    one_pos = []
    if write:
        fout= codecs.open(output, "w", encoding='utf-8')

    with codecs.open(input, "r", encoding='utf-8') as fin:
        for line in fin:
            # if line.startswith("#"):
            #     continue
            if line == "" or line == "\n":
                if write:
                    fout.write("\n")
                pos_tags.append(one_pos)
                one_pos = []
            else:
                info = line.strip().split("\t")
                feats = info[5].split("|")
                new_feats = []
                isPos = False
                for feat in feats:
                    key_value = feat.split("=")
                    if key_value[0] == "POS":
                        info[3] = key_value[-1]
                        one_pos.append(info[3])
                        isPos = True
                    else:
                        new_feats.append(feat)

                if not isPos:
                    info[3] = "_"
                    one_pos.append(info[3])
                if len(new_feats) == 0:
                    info[5] = "_"
                else:
                    info[5] = "|".join(new_feats)
                if write:
                    fout.write("\t".join(info) + "\n")

    return pos_tags


def checkForAttribute(input, attribute, max_lines=0):

    with codecs.open(input, "r", encoding='utf-8') as fin:
        num_lines = 0
        
        for line in fin:
            if line.startswith("#"):
                continue

            if line == "" or line == "\n":
                num_lines += 1
                if max_lines > 0 and num_lines == max_lines:
                    break
                
            else:
                info = line.strip().split("\t")
                if info[5] != "_":
                    feats = info[5].split("|")
                    for feat in feats:
                        key_value = feat.split("=")
                        if key_value[0] == attribute:
                            return True

    return False
                            

def createPOSFiles(input, output, gold_output, attribute, max_lines=0):
    if gold_output is not None:
        fgold = codecs.open(gold_output, "w", encoding='utf-8')
    with codecs.open(input, "r", encoding='utf-8') as fin, codecs.open(output, "w", encoding='utf-8') as fout:
        num_lines = 0
        for line in fin:
            # if line.startswith("#"):
            #     continue

            if line == "" or line == "\n":
                fout.write("\n")
                if gold_output is not None:
                    fgold.write("\n")
                num_lines += 1
                if max_lines > 0 and num_lines == max_lines:
                    break

            else:
                info = line.strip().split("\t")
                if info[5] == "_":
                    fout.write(info[1] + "\t" + "_" +"\n")
                    if gold_output is not None:
                        #info[5] = attribute+ "=" + info[5]
                        fgold.write("\t".join(info) + "\n")
                else:
                    feats = info[5].split("|")
                    isAttribute=False
                    for feat in feats:
                        key_value = feat.split("=")
                        if key_value[0] == attribute:
                            fout.write( info[1] + "\t" +key_value[-1] +"\n" )
                            info[5] = key_value[-1]
                            if gold_output is not None:
                                #info[5] = attribute + "=" + info[5]
                                fgold.write("\t".join(info) + "\n")
                            isAttribute = True
                            break

                    if not isAttribute:
                        fout.write(info[1] + "\t" + "_" + "\n")
                        if gold_output is not None:
                            info[5] = "_"
                            fgold.write("\t".join(info) + "\n")


if __name__ == "__main__":

    train_file = args.train.split("/")[-1]
    udmap_train_file = args.input + "/" + "udmap_" + train_file

    dev_file = args.dev.split("/")[-1]
    udmap_dev_file = args.input + "/" + "udmap_" + dev_file

    test_file = args.test.split("/")[-1]
    udmap_test_file = args.input + "/" + "udmap_" + test_file

    if not os.path.exists(udmap_train_file) and not os.path.exists(udmap_dev_file) and not os.path.exists(udmap_test_file):
        print("Creating udmap files and pretrain_pos files...")
        inverseAttributes = parseInverseAttributes(args.attributes)

        output_file = args.input + "/" + "udmap_" + train_file
        mapTaskData.convert(args.train, output_file, inverseAttributes)
        output_file = args.input + "/" + "pretrain_pos_udmap_" + train_file
        convert(udmap_train_file, output_file)

        file = args.dev.split("/")[-1]
        output_file = args.input + "/" + "udmap_" + dev_file
        mapTaskData.convert(args.dev, output_file, inverseAttributes)
        output_file = args.input + "/" + "pretrain_pos_udmap_" + dev_file
        convert(udmap_dev_file, output_file)

        file = args.test.split("/")[-1]
        output_file = args.input + "/" + "udmap_" + test_file
        mapTaskData.convert(args.test, output_file, inverseAttributes)
        output_file = args.input + "/" + "pretrain_pos_udmap_" + test_file
        convert(udmap_test_file, output_file)

    else:
        print("udmap files exist, creating pretrain_pos files ... ")
        output_file = args.input + "/" + "pretrain_pos_udmap_" + train_file
        if not os.path.exists(output_file):
            convert(udmap_train_file, output_file)

        output_file = args.input + "/" + "pretrain_pos_udmap_" + dev_file
        if not os.path.exists(output_file):
            convert(udmap_dev_file, output_file)

        output_file = args.input + "/" + "pretrain_pos_udmap_" + test_file
        if not os.path.exists(output_file):
            convert(udmap_test_file, output_file)

    attributes = parseAttributes(args.attributes)
    for attribute in attributes.keys():
        if attribute != "POS":
            continue
        
        print("Checking for attribute", attribute)
        input_file = args.input + "udmap_" + args.train.split("/")[-1]
        is_attribute = checkForAttribute(input_file, attribute, args.max_lines)

        if is_attribute:
            print"Creating train and dev file for", attribute, "attribute"
            output_file = args.input + attribute + "_train.conll"
            gold_output_file = args.input + "udmap_" + attribute + "_gold_train.conllu"
            createPOSFiles(input_file, output_file, None, attribute, args.max_lines)

            input_file = args.input + "udmap_" + args.dev.split("/")[-1]
            output_file = args.input + attribute + "_dev.conll"
            gold_output_file = args.input +  attribute + "_gold_dev.conllu"
            createPOSFiles(input_file, output_file, gold_output_file, attribute)

            input_file = args.input + "udmap_" + args.test.split("/")[-1]
            output_file = args.input + attribute + "_test.conll"
            gold_output_file = args.input + "udmap_" + attribute + "_gold_test.conllu"
            createPOSFiles(input_file, output_file, None, attribute)

        else:
            print("Attribute not found")
