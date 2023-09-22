import argparse
import numpy as np


def extract_neg(neg_sad, out_file):
    with open(neg_sad, 'r') as f:
        neg_data = f.readlines()
    neg_data = neg_data[1:]
    neg_list = []
    for i in neg_data:
        scores = [float(val) for val in i.split()[-1].split(",")]
        neg_list.append(scores)
    neg_array = np.array(neg_list)
    neg_array = np.transpose(neg_array)
    np.save(out_file, neg_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Saves sad-score array of negative non-functional SNP set to .npy file')
    parser.add_argument('-n', '--neg_sad', default = "/media/hmg/InternalHDD/anil/neurally/neg_vars_sadscores.txt", help='negative variant set sad-score file')
    parser.add_argument('-o', '--out_file', default="neg_sadscores", help='.npy file having sad-score array of negative set SNPs')
    args = parser.parse_args()
    
    print("Writing sad-score array of negative non-functional SNPs to .npy file...")
    extract_neg(args.neg_sad, args.out_file)

