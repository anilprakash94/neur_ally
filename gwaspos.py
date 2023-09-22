import os, re, argparse


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
            var_dict[line[2]] = [chrom, line[1], line[3], line[4]]
    return var_dict
    

def extract_vars(gwas_file, var_dict, exon_dict):
    #keywords belonging to neurological traits
    traits = ["alzheimer", "epilepsy", "multiple sclerosis", "parkinson", "autism", "attention deficit", "schizophrenia", "bipolar", "major depressive"]
    keywords = []
    for word in traits:
        keywords.append(word)
        keywords.append(word.capitalize())
    gwas_var = set()
    with open(gwas_file, 'r') as f:
        gwas_data = f.readlines()
    gwas_data = gwas_data[1:]
    for i in gwas_data:
        line = i.split("\t")
        trait_info = line[7]
        snp = line[20].split("-")[0]
        for j in keywords:
            if j in trait_info:
                if snp in var_dict:
                    chrom, pos, ref, alt = var_dict[snp]
                    snp_info = chrom + ":" + pos + ":" + ref + ":" + alt
                    if snp_info in gwas_var:
                        break
                    else:
                        exon_var = filter_var(chrom, pos, exon_dict)
                        if exon_var == False:
                            gwas_var.add(snp_info)
                break
    return gwas_var


def save_vars(gwas_var, out_file):
    with open(out_file, 'w') as output:
        for i in gwas_var:
            line = i.split(":")         
            output.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + line[3] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Outputs neurologic GWAS variant coordinates to file')
    parser.add_argument('-g', '--gwas_file', default = "/home/hmg/gwas_catalog_all_v1.0", help='gwas catalog all dataset')
    parser.add_argument('-o', '--out_file', default="gwas_pos_vars.txt", help='output file having gwas catalog SNP coordinates')
    parser.add_argument('--dbsnp_file', default = 'common_all_20180418.vcf', help='dbsnp vcf file with common variants')
    parser.add_argument('-e', '--exon_file', default = "gencode_exons_modif_canonical.bed", help='bed file with canonical exon coordinates according to hg38 build')
    args = parser.parse_args()

    #raises exception if the gwas catalog file does not exist in the current location
    
    if not os.path.isfile(args.gwas_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.gwas_file)

    print("Loading exon coordinates from file...")
    exon_dict = read_bed(args.exon_file)
    
    print("Extracting common variant data from dbSNP database...")
    var_dict = extract_vcf(args.dbsnp_file)
    
    print("Extracting neurological GWAS SNP coordinates to set...")
    gwas_var = extract_vars(args.gwas_file, var_dict, exon_dict)
    
    print("Writing SNPs to file...")
    save_vars(gwas_var, args.out_file)

