import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text


def extract_vcf(dbsnp_file):
    var_dict = {}
    with open(dbsnp_file) as f:
        var_data = f.readlines()
    header_num = 0
    for i in var_data:
        if i.startswith("#"):
            header_num += 1
        else:
            break
    var_data = var_data[header_num:]
    for j in var_data:
        line = j.split()
        if (len(line[3]) == 1 and len(line[4]) == 1):
            chrom = "chr" + line[0]
            snp_info = chrom + ":" + line[1]
            var_dict[snp_info] = line[2]
    return var_dict


def extract_names(dataset_files):
    encfiles = []
    with open(dataset_files, 'r') as f:
        file = f.readlines()
    for ele in file:
        enc_name = "_".join(ele.split("_")[:-1])
        encfiles.append(enc_name)
    return encfiles


def create_df(eval_file, encfiles):
    with open(eval_file, 'r') as f:
        eval_data = f.readlines()
    eval_list = []
    for i in range(1, len(eval_data)):
        eval_list.append([eval_data[i].split()[0].replace('chr', '')] + [int(eval_data[i].split()[1])] + eval_data[i].split()[-1].split(","))
    cols = ['chrom', 'coordinate'] + encfiles
    eval_df = pd.DataFrame(eval_list, columns=cols)
    eval_df['min_eval'] =  eval_df.iloc[:,2:].astype(float).min(axis=1)
    eval_df['min_eval'] = eval_df['min_eval'].replace(0,1e-07)
    eval_df['min_eval'] = -np.log10(eval_df['min_eval'])
    chr_order = list(map(str, range(1,23))) + ['X', 'Y']
    eval_df['chrom'] = pd.Categorical(eval_df['chrom'], chr_order)
    eval_df.sort_values(by=['chrom','coordinate'],inplace=True) 
    eval_df['ind'] = range(len(eval_df))
    return eval_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots manhattan plot of SNPs with significant e-values')
    parser.add_argument('-e', '--eval_file', default = "/media/hmg/InternalHDD/anil/neurally/asd_pos_vars_evalue.txt", help='snp e-value file')
    parser.add_argument('--thresh', type=float, default=1e-05, help='e-value threshold')
    parser.add_argument('-d', '--dataset_files', dest = "dataset_files" , default="datasets/encfiles.txt", help='text file containing names of encode dataset files')
    parser.add_argument('--dbsnp_file', default = 'common_all_20180418.vcf', help='dbsnp vcf file with common variants')
    parser.add_argument('-p', '--plot_file', default="asd_pos_vars_manhattan.png", help='heatmap of e-values')
    args = parser.parse_args()
    
    print("Extracting common variant data from dbSNP database...")
    var_dict = extract_vcf(args.dbsnp_file)
        
    print("Extracting encode filenames...")
    encfiles = extract_names(args.dataset_files)
    
    print("Extracting SNPs of significant e-values...")
    eval_df = create_df(args.eval_file, encfiles)
    plot = sns.relplot(data=eval_df, x='ind', y='min_eval', aspect=4, hue='chrom', palette = 'bright', legend=False, height=6)
    threshold = -np.log10(args.thresh)
    ax1 = plot.axes.flatten()[0]
    ax1.axhline(threshold, ls='--')
    chrom_df = eval_df.groupby('chrom')['ind'].median()
    plot.ax.set_xlabel('Chromosome')
    plot.ax.set_ylabel('-log10(min e-value)')
    plot.ax.set_xticks(chrom_df);
    plot.ax.set_xticklabels(chrom_df.index)
    bottom = ax1.get_ylim()[0]
    plot.set(ylim=(bottom,9))
    plot.fig.suptitle('Manhattan plot')
    #label significant SNPs
    sig_df = eval_df[eval_df['min_eval'] >= threshold]
    texts = []
    for var in range(len(sig_df)):
        snp_info = 'chr' + sig_df.iloc[var]['chrom'] + ":" + str(sig_df.iloc[var]['coordinate'])
        if snp_info in var_dict:
            label_txt = var_dict[snp_info]
        else:
            label_txt = snp_info
        x, y = sig_df.iloc[var]['ind'], sig_df.iloc[var]['min_eval']
        texts.append(ax1.annotate(label_txt,(x,y)))
    if texts != []:
        adjust_text(texts)
    plot.savefig(args.plot_file, dpi=300)
    

