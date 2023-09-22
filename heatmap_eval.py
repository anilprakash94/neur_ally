import argparse
import pandas as pd
import numpy as np
import seaborn as sns

def extract_names(dataset_files):
    encfiles = []
    with open(dataset_files, 'r') as f:
        file = f.readlines()
    for ele in file:
        enc_name = "_".join(ele.split("_")[:-1])
        encfiles.append(enc_name)
    return encfiles


def create_df(eval_file, thresh, encfiles):
    with open(eval_file, 'r') as f:
        eval_data = f.readlines()
    eval_list = []
    for i in range(1, len(eval_data)):
        eval_list.append([":".join(eval_data[i].split()[:2])] + eval_data[i].split()[-1].split(","))
    cols = ['SNP coordinates'] + encfiles
    eval_df = pd.DataFrame(eval_list, columns=cols)
    eval_df.set_index('SNP coordinates', inplace=True)
    eval_df = eval_df.astype(float)
    bool_df = eval_df<=thresh
    bool_df = bool_df.any(axis=1)
    eval_df = eval_df.loc[bool_df[bool_df == True].index]
    eval_df.replace(0,1e-07, inplace=True)
    eval_df = -np.log10(eval_df)
    return eval_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots heatmap of SNPs with significant e-values')
    parser.add_argument('-e', '--eval_file', default = "/media/hmg/InternalHDD/anil/neurally/brain_eqtl_pos_vars_evalue.txt", help='snp e-value file')
    parser.add_argument('--thresh', type=float, default=1e-05, help='e-value threshold')
    parser.add_argument('-d', '--dataset_files', dest = "dataset_files" , default="datasets/encfiles.txt", help='text file containing names of encode dataset files')
    parser.add_argument('-p', '--plot_file', default="brain_eqtl_pos_vars_evalue.png", help='heatmap of e-values')
    args = parser.parse_args()
    
    print("Extracting encode filenames...")
    encfiles = extract_names(args.dataset_files)
    
    print("Extracting SNPs of significant e-values...")
    eval_df = create_df(args.eval_file, args.thresh, encfiles)
    #print(len(eval_df))
    print("Plotting e-values of input SNPs...")
    width = 100
    height = 80
    sns.set(rc = {'figure.figsize':(width,height)})
    sns_plot = sns.heatmap(eval_df)
    sns_plot.set_title('E-value Heatmap',fontdict= { 'fontsize': 100, 'fontweight':'bold'})
    sns_plot.figure.axes[-1].tick_params(labelsize=60)
    sns_plot.set_xlabel('Epigenomic labels', fontsize=50, fontdict={'weight': 'bold'})
    sns_plot.set_ylabel(None)
    sns_plot.set_xticklabels([])
    sns_plot.set_yticklabels([])
    sns_plot.figure.savefig(args.plot_file, dpi=300)    

