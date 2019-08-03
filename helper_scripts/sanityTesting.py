import codecs
import argparse, os
import evaluate_2019_task2

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/UpdatedSIG/")
parser.add_argument("--test_dir", type=str, default="/Users/aditichaudhary/Documents/CMU/sigmorph_data/task2/")
args = parser.parse_args()

def loadTest(input):
    with codecs.open(input, "r", encoding='utf-8') as fin:
        test_lines = []
        one_line = []
        for line in fin:
            if line.startswith("#"):
                continue
            if line == "" or line == "\n":
                test_lines.append(one_line)
                one_line = []
            else:
                info = line.strip().split("\t")
                one_line.append(info[1])

        if len(one_line) > 0:
            test_lines.append(one_line)
    return test_lines


def loadOutput(input):
    with codecs.open(input, "r", encoding='utf-8') as fin:
        test_lines, test_feats, test_lemmas = [], [], []
        one_line, one_feat, one_lemma = [], [], []
        for line in fin:
            if line.startswith("#"):
                continue
            if line == "" or line == "\n":
                test_lines.append(one_line)
                test_feats.append(one_feat)
                test_lemmas.append(one_lemma)
                one_line, one_feat, one_lemma = [], [], []
            else:
                info = line.strip().split("\t")
                one_line.append(info[1])
                one_feat.append(info[5])
                one_lemma.append(info[2])

        if len(one_line) > 0:
            test_lines.append(one_line)
            test_feats.append(one_feat)
            test_lemmas.append(one_lemma)
    return test_lines, test_feats, test_lemmas

if __name__ == "__main__":

    for [path, dirs, files] in os.walk(args.input_dir):
        for dir in dirs:
            outdir = path + "/" + dir
            if dir != "UD_Marathi-UFAL":
                continue
            print("Testing Dir, ", outdir)
            for [p, _, files] in os.walk(outdir):
                test_file =  outdir + "/" + files[0]

                test_lines = loadTest(test_file)

                test_dir = args.test_dir  + "/" +  dir
                for [_,_, files] in os.walk(test_dir):
                    for file in files:
                        if "-um-test" in file and not file.startswith("pretrain") and not file.startswith("udmap"):
                            output_file = test_dir + "/" + file
                            break

                output_lines, output_feats, output_lemmas = loadOutput(output_file)

                print("Checking length of lines")
                assert len(test_lines) == len(output_lines)
                assert len(output_lines) == len(output_feats)
                assert len(output_feats) == len(output_lemmas)
                print(output_file, test_file)
                (lemma_acc, lemma_f1, morph_acc, morph_f1,_,_) = evaluate_2019_task2.main(ref_file=output_file, out_file=test_file)
                print(lemma_acc, lemma_f1, morph_acc, morph_f1)

                '''
                print("Checking length of tokens ")
                write=False
                for index in range(len(output_lines)):
                    test_line = test_lines[index]
                    output_line = output_lines[index]
                    output_feat = output_feats[index]
                    output_lemma = output_lemmas[index]

                    assert len(test_line) == len(output_line)
                    assert len(output_feat) == len(output_line)
                    assert len(output_lemma) == len(output_line)

                    for token_index in range(len(output_feat)):
                        if ";" in output_feat[token_index] and "_" in output_feat[token_index]:
                            print("ERROR in feats!")
                            exit(-1)
                        if "" == output_feat[token_index] or " " == output_feat[token_index]:
                            print("ERROR! in feats, field empty")
                            exit(-1)

                        # if output_lemma[token_index] == "" or output_lemma[token_index] == " " or output_lemma[token_index] == "_":
                        #     print("ERROR! in lemma")
                        #     exit(-1)
                        #     output_lemma[token_index] = test_line[token_index]
                        #     write=True
                if write:
                    with codecs.open(args.output  + ".fixed.conllu", "w", encoding='utf-8') as fout:
                        print("Post fixing file", args.output  + ".fixed.conllu")
                        for index in range(len(output_lines)):
                            output_feat = output_feats[index]
                            for token_index in range(len(output_feat)):
                                info = ["_" for _ in range(10)]
                                info[0] = str(token_index + 1)
                                info[1] = test_lines[index][token_index]
                                info[2] = output_lemmas[index][token_index]
                                info[5] = output_feat[token_index]
                                fout.write("\t".join(info) + "\n")
                            fout.write("\n")
                print("All Good!")
                '''
