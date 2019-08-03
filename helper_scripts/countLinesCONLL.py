import codecs
import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("--input", help="Folder with the raw text data",
                        default=None,
                        type=str)

args = arg_parser.parse_args()
print("Args used for this run:")
print(args)

def count(input):
	with codecs.open(input,"r",encoding='utf-8') as fin:
		index = 0
		for line in fin:
			if line.strip() == "":
				index = index +1

	index = index + 1

	return index
if __name__ == "__main__":
	index = count(args.input)
	print(index)
		
