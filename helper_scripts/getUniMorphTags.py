import codecs,os, argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="/Users/aditichaudhary/Documents/CMU/sigmorph_data/task2/")
parser.add_argument("--output", type=str, default="/Users/aditichaudhary/Documents/CMU/sigmorph_data/unimorph_tagset.txt")
args = parser.parse_args()

def findTags():
    tagset = set()
    for [path, dirs, _] in os.walk(args.dir):
        for dir in dirs:
            dir = path + "/" + dir
            for [p, _, files] in os.walk(dir):
                for file in files:
                    if file.endswith(".conllu") and not file.startswith("mapped"):
                        filename = p + "/" + file
                        print("Processing : {0}".format(filename))


                        with codecs.open(filename, "r", encoding='utf-8') as fin:

                            one_line = []
                            one_feats = []
                            for line in fin:
                                if line.startswith("#"):
                                    continue

                                if line == "" or line == "\n":
                                   continue

                                else:
                                    info = line.strip().split("\t")
                                    one_line.append(info[1])
                                    one_feats.append(info[5])
                                    values = info[5].split(";")
                                    for value in values:
                                        tagset.add(value)


    with codecs.open(args.output, "w", encoding='utf-8') as fout:
        for tag in tagset:
            fout.write(tag + "\n")

if __name__ == "__main__":
    findTags()