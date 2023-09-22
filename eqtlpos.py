import os, re, argparse
import numpy as np


#loading all significant variant-gene pair files in the folder

def extract_files(dataset_path):
    gtex_files = []
    file_names = os.listdir(dataset_path)
    for names in file_names:
        if names.endswith(".txt"):
            gtex_files.append(names)
    return gtex_files


def read_bed(exon_file):
    exon_dict = {}
    with open(exon_file) as f:
        file_data = f.readlines()
    for i in file_data:
        line = i.split()
        if line[0] not in exon_dict:
            exon_dict[line[0]] = []
        exon_dict[line[0]].append(line[1:])
    return exon_dict


def filter_var(chrom, pos, exon_dict):
    exon_var = False
    for ele in exon_dict[chrom]:
        if float(pos) > float(ele[0]) and float(pos) <= float(ele[1]):
            exon_var = True
            break
    return exon_var


def extract_vars(dataset_path, gtex_files, exon_dict):
    eqtl_set = set()
    for path in gtex_files:
        print("Extracting eQTLs of sample: ", path)
        if dataset_path.endswith("/"):
            filepath = dataset_path + path
        else:
            filepath = dataset_path + "/" + path
        with open(filepath,'r') as f:
            txt_file = f.readlines()
        txt_file = txt_file[1:]
        snp_list = []
        pval_list = []
        for i in txt_file:
            variant = i.split("\t")[0]
            chrom, pos, ref, alt, build = variant.split("_")
            pval = float(i.split("\t")[6])
            if (len(ref) == 1 and len(alt) == 1):
                snp_info = chrom+":"+pos+":"+ref+":"+alt
                if snp_info in eqtl_set:
                    pass
                else:
                    snp_list.append(snp_info)
                    pval_list.append(pval)
        sort_idx = np.argsort(pval_list)
        sorted_snp = np.array(snp_list)[sort_idx]
        var_num = 0
        for var_ele in sorted_snp:
            if var_num == 1000:
                break
            chrom , pos = var_ele.split(":")[:2]
            exon_var = filter_var(chrom, pos, exon_dict)
            if (exon_var == False and var_ele not in eqtl_set):
                var_num += 1
                eqtl_set.add(var_ele)
    return eqtl_set


def save_vars(eqtl_set, out_file):
    with open(out_file, 'w') as output:
        for i in eqtl_set:
            line = i.split(":")         
            output.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + line[3] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Outputs top neurologic eQTL variant coordinates to file')
    parser.add_argument('-d', '--dataset_path', dest = "dataset_path" , default="/home/hmg/eqtl_data/brain", help='path of folder containing eqtl significant variant-gene pair files in .txt format')
    parser.add_argument('-o', '--out_file', default="brain_eqtl_pos_vars.txt", help='output file having eQTL variant coordinates')
    parser.add_argument('-e', '--exon_file', default = "gencode_exons_modif_canonical.bed", help='bed file with canonical exon coordinates according to hg38 build')
    args = parser.parse_args()
    
    print("Extracting all filenames with significant variant-gene pairs...")
    gtex_files = extract_files(args.dataset_path)
    
    print("Extracting exon coordinates to file...")
    exon_dict = read_bed(args.exon_file)
    
    print("Adding top significant eQTL coordinates to a set...")
    eqtl_set = extract_vars(args.dataset_path, gtex_files, exon_dict)
    
    print("Writing eQTL variant coordinates to file...")
    save_vars(eqtl_set, args.out_file)


