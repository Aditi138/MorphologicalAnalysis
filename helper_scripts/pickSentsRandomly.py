import codecs
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_train", type=str)
parser.add_argument("--pos_train", type=str)
parser.add_argument("--out_train", type=str)
parser.add_argument("--out_pos", type=str)
parser.add_argument("--count", type=int)
args = parser.parse_args()

def readInput(input):
    input_lines = []
    one_line = []
    with codecs.open(input, "r", encoding='utf-8') as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            if line == "" or line == "\n":
                input_lines.append(one_line)
                one_line = []
            else:
                one_line.append(line)
        if len(one_line) > 0:
            input_lines.append(one_line)
    return input_lines

input_lines = readInput(args.input_train)
pos_lines = readInput(args.pos_train)
#print(input_lines[0], pos_lines[0])
assert len(input_lines) == len(pos_lines)
output_lines = []
output_pos = []
idx = np.random.choice(len(input_lines), args.count, replace=False)
with codecs.open(args.out_train, "w", encoding='utf-8') as fout, codecs.open(args.out_pos, "w", encoding='utf-8') as fpos:
    for i in idx:
        output_line = input_lines[i]
        output_pos = pos_lines[i]
        if len(output_line) != len(output_pos):
             print(i, output_line, output_pos)
             print(len(output_line), len(output_pos))
        assert len(output_line) == len(output_pos)
        for token, token_pos in zip(output_line, output_pos):
            fout.write(token)
            fpos.write(token_pos)
        fout.write("\n")
        fpos.write("\n")
