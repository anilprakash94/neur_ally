import argparse, random
import numpy as np


def extract_pos(pos_sad, neg_file, out_file, pos_num):
    neg_data = np.load(neg_file, mmap_mode='r')
    with open(pos_sad, 'r') as f:
        pos_data = f.readlines()
    pos_data = pos_data[1:]
    #if pos_num is 0, equal number of positive and negative variants will be used.
    if pos_num == 0:
        pos_num = len(pos_data)
    header = ["Chr", "Pos", "Ref", "Alt", "E_value"]
    variant_num = 0
    with open(out_file, 'w') as output:
        output.write("\t".join(header) + "\n")
        for var in pos_data:
            variant_num += 1
            print("Calculating E-Value for variant no. ", variant_num)  
            score_list = var.split()[-1].split(",")
            score_array = np.array(score_list, dtype=float)
            score_array = np.expand_dims(score_array, axis=1)
            #10 times sub-sampling from the negative snp sad score file.
            sub_sample = 10
            snp_eval = []
            for num in range(sub_sample):
                neg_num = random.sample(range(neg_data.shape[1]), pos_num)
                sub_data = neg_data[:, neg_num]
                evalue_arr = sub_data > score_array
                snp_eval.append(np.mean(evalue_arr, axis=1).tolist())
            snp_eval = np.mean(np.array(snp_eval), axis=0)
            snp_info = var.split()[:4] + [",".join(str(x) for x in snp_eval)]
            output.write("\t".join(snp_info) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Outputs e-value of positive SNPs to file')
    parser.add_argument('-p', '--pos_sad', default = "/media/hmg/InternalHDD/anil/neurally/reg_snps_sadscores.txt", help='postive variant set sad-score text file')
    parser.add_argument('-n', '--neg_file', default = "/media/hmg/InternalHDD/anil/neurally/neg_sadscores.npy", help='negative variant set sad-score .npy file')
    parser.add_argument('--pos_num', default = 0, type=int, help='number of positive samples to be considered for evalue calculation')    
    parser.add_argument('-o', '--out_file', default="reg_snps_evalue.txt", help='output file having e-values of positive set variants')
    args = parser.parse_args()
    
    print("Writing e-values of positive SNPs to file...")
    extract_pos(args.pos_sad, args.neg_file, args.out_file, args.pos_num)

