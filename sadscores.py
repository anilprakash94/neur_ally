import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import errno
import numpy as np
import pandas as pd
import math
import argparse   
from pyfaidx import Fasta
import re

from neurally_train import *


def read_mutation(mutation_file):
    with open(mutation_file, 'r') as f:
        data = f.readlines()
    mut_list = []
    for i in data:
        if i.startswith("#"):
            pass
        else:
            line = i.split()
            line = line[0:4]
            mut_list.append(line)
    return mut_list


def load_weights(args, test_model, model_name, epoch_num):
    print("Loading weights from file...")
    model_dir = args.model_dir
    dir_path = weights_filepath(model_dir+"/"+model_name)
    file_names = os.listdir(dir_path)
    idx_files = []
    for names in file_names:
        if names.endswith(".index"):
            idx_files.append(names)
    for idx in idx_files:
        if int(re.search('weights.(.+?)-', idx).group(1)) == epoch_num:
            weights_file = idx[0:-6]
    weights_path = dir_path + "/" + weights_file
    status = test_model.load_weights(weights_path).expect_partial() 
    print("Asserting matching of weights...")
    status.assert_existing_objects_matched()
    return test_model


#input mutation coordinates should be 1-based
def var_mut(mut_list, ref_genome, epi_model, args, encfiles, peak_len=200):
    seq_len = args.seq_len
    var_effect = {}
    var_num = 0
    for i in mut_list:
        var_num += 1
        print("Predicting effects of Variant No. :", var_num)
        chr_name, mut_pos, ref_allele, alt_allele = i
        if not chr_name.startswith("chr"):
            chr_name = "chr" + chr_name
        #chromosomal position of mutation 
        mut_pos = int(mut_pos)
        #position of starting nucleotide of 200bp bin 
        if mut_pos%peak_len == 0:
            left_nuc = peak_len * ((mut_pos-1)//peak_len)
        else:
            left_nuc = peak_len * (mut_pos//peak_len)
        #position of ending nucleotide of 200bp bin 
        right_nuc = left_nuc + peak_len
        bin_left = ref_genome[chr_name][left_nuc:mut_pos].seq
        bin_right = ref_genome[chr_name][mut_pos:right_nuc].seq
        #nucleotide sequence of 200bp bin with reference allele
        bin_seq = bin_left + bin_right
        alt_left = ref_genome[chr_name][left_nuc:mut_pos-1].seq
        alt_right = ref_genome[chr_name][mut_pos:right_nuc].seq
        #nucleotide sequence of 200bp bin with alternate allele
        alt_bin = alt_left + alt_allele + alt_right
        #predicts epigenetic labels of 200bp sequence bins around the mutation
        range_start = (right_nuc+(peak_len//2)) - seq_len
        range_end = right_nuc-(peak_len//2)
        final_score = []
        for bin_range in range(range_start, range_end, 200):
            #position of starting nucleotide of input sequence with length = args.seq_len
            left_flank = bin_range
            #position of ending nucleotide of input sequence
            right_flank = bin_range + seq_len
            start = max(0,left_flank)
            left_seq = ref_genome[chr_name][start:left_nuc].seq
            right_seq = ref_genome[chr_name][right_nuc:right_flank].seq
            left_flanklen = left_nuc - left_flank
            right_flanklen = right_flank - right_nuc
            pad_left, pad_right = "", ""
            if len(left_seq) < left_flanklen:
                pad_left = 'N' * (left_flanklen - len(left_seq) ) 
            if len(right_seq) < right_flanklen:
                pad_right = 'N' * (right_flanklen - len(right_seq) )
            #nucleotide sequence of the input region with reference allele
            ref_seq = pad_left + left_seq + bin_seq + right_seq + pad_right
            #nucleotide sequence of the input region with alternate allele
            alt_seq = pad_left + left_seq +  alt_bin + right_seq + pad_right
            variant_seq = [ref_seq, alt_seq]
            pred_list = []
            for seq_data in variant_seq:
                input_array = vectorization(seq_data)
                input_array = np.expand_dims(input_array,0)
                out_arr = epi_model(input_array).numpy()
                pred_list.append(out_arr)
            #predicts SNP Activity Difference (SAD) score between label prediction probabilities of input sequence with reference allele vs alternate allele.
            sad_score = np.absolute(pred_list[0] - pred_list[1])
            sad_score = sad_score.flatten().tolist()
            final_score.append(sad_score)
        final_score = np.array(final_score)
        mean_score = np.mean(final_score, axis=0).tolist()
        mean_score = ",".join([str(score) for score in mean_score])
        var_effect[":".join(value for value in i)] = [mean_score]
    return var_effect


def save_results(var_effect, args):
    header = ["Chr", "Pos", "Ref", "Alt", "SAD_score"]
    out_path = args.var_output
    with open(out_path, 'w') as output:
        output.write("\t".join(header) + "\n")
        for x,y in var_effect.items():
            res = x.split(":") + y
            output.write("\t".join(val for val in res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Variant effect prediction using in-silico mutagenesis')
    parser.add_argument("--epoch_num", type=int, help="epoch number of weights file having minimum validation loss")
    parser.add_argument('--dim', type=int, default=128, help='dimensions of input after embedding in the epigenomic model')
    parser.add_argument('--seq_len', type=int, default=2000, help='total flanking+bin sequence length for each input')
    parser.add_argument('--heads', type=int, default=4, help='number of multi head attention heads')
    parser.add_argument('-d', '--dataset_files', default="datasets/encfiles.txt", help='text file containing names of encode dataset files')
    parser.add_argument('--mutation_file', default="/media/hmg/InternalHDD/anil/neurally/neg_vars.txt", help='text or vcf file having non-coding mutations to be tested')
    parser.add_argument('--var_output', default="/media/hmg/InternalHDD/anil/neurally/neg_vars_sadscores.txt", help='specify output text file for writing prediction results')
    parser.add_argument('--model_dir', default = "Models", help='specify the output folder name for saving the model')
    parser.add_argument('-r', '--ref_fasta', default = "datasets/hg38.fa", help='specify the reference genome fasta file')
    parser.add_argument('--model_name', default="neurally", help='specify the name of the model under study')
    args, unknown = parser.parse_known_args()
    
    
    #raises exception if the reference genome fasta file does not exist in the current location
    
    if not os.path.isfile(args.ref_fasta):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.ref_fasta)
    
    print("Loading mutations from the input query file...")
    mut_list = read_mutation(args.mutation_file)
    
    print("Saving reference genome to variable...")
    ref_genome = Fasta(args.ref_fasta)
    
    print("Extracting filenames...")
    encfiles = extract_names(args)
    label_num = len(encfiles)
    #create positional encoding array
    pos = pos_encode(args.seq_len, args.dim)
    print("Initializing the epigenomic model...")
    epi_model = Modelsubclass(args, label_num, pos)
    
    #create one sample batch for running model once
    x_single = np.random.uniform(size=(1,args.seq_len))
    
    epi_model(input_data=x_single)
    
    epi_model = load_weights(args, epi_model, args.model_name, args.epoch_num)
    
    print("Predicting regulatory changes for reference and mutated genome sequences...")
    var_effect = var_mut(mut_list, ref_genome, epi_model, args, encfiles, peak_len=200)
    print("Writing variant effects to file...")
    save_results(var_effect, args)
    
    
    
