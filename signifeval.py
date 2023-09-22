import argparse
import numpy as np

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


def sort_var(eval_file, var_dict, out_file, thresh, dataset_files):
    encfiles = []
    with open(dataset_files, 'r') as f:
        file_names = f.readlines()
    for j in file_names:
        encfiles.append(j.strip())
    with open(eval_file, 'r') as f:
        var_data = f.readlines()
    var_data = var_data[1:]
    header = ["Chr", "Pos", "Ref", "Alt", "SNP", "Signif. labels", "Signif. scores"]
    with open(out_file, 'w') as output:
        output.write(",".join(header) + "\n")
        for ele in var_data:
            chrom, pos, ref, alt = ele.split()[:4]
            snp_info = chrom + ":" + pos
            rsid = "-"
            if snp_info in var_dict:
                rsid = var_dict[snp_info]
            score_list = ele.split()[-1].split(",")
            score_array = np.array(score_list, dtype=float)
            score_idx = np.argwhere(score_array <= thresh).flatten()
            if len(score_idx) != 0:
                sig_scores = np.array(score_list)[score_idx].tolist()
                sig_scores = "\t".join(sig_scores)
                sig_labels = np.array(encfiles)[score_idx].tolist()
                sig_labels = "\t".join(sig_labels)
                sig_res = [chrom, pos, ref, alt, rsid, sig_labels, sig_scores]
                output.write(",".join(sig_res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Writes significant e-values of positive SNPs to file')
    parser.add_argument('-e', '--eval_file', default = "/media/hmg/InternalHDD/anil/neurally/asd_pos_vars_evalue.txt", help='e-value file of variants')
    parser.add_argument('-d', '--dataset_files', dest = "dataset_files" , default="datasets/encfiles.txt", help='text file containing names of encode dataset files')
    parser.add_argument('--thresh', type=float, default=1e-05, help='e-value threshold')
    parser.add_argument('--dbsnp_file', default = 'common_all_20180418.vcf', help='dbsnp vcf file with common variants')
    parser.add_argument('-o', '--out_file', default="asd_pos_vars_signif.txt", help='output file having sorted e-values of input variants')
    args = parser.parse_args()
    
    print("Extracting common variant data from dbSNP database...")
    var_dict = extract_vcf(args.dbsnp_file)
    
    print("Writing sorted e-values of SNPs to file...")
    sort_var(args.eval_file, var_dict, args.out_file, args.thresh, args.dataset_files)

