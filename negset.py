import argparse, random
from liftover import get_lifter


def extract_pos(gwas_file, eqtl_file):
    pos_var = set()
    with open(gwas_file, 'r') as f:
        gwas_data = f.readlines()
    for i in gwas_data:
        chrom, pos = i.split()[:2]
        pos_var.add(chrom+":"+pos)
    with open(eqtl_file, 'r') as f:
        eqtl_data = f.readlines()
    for i in eqtl_data:
        chrom, pos = i.split()[:2]
        pos_var.add(chrom+":"+pos)
    return pos_var



#using liftover for snp coordinate conversion can be problematic due to conversion-unstable positions (CUPs).
#filter CUPs while converting variant positions between builds.
#CUP coordinates can be downloaded from "https://github.com/cathaloruaidh/genomeBuildConversion".

def read_bed(cup_file, cup_dict):
    with open(cup_file) as f:
        file_data = f.readlines()
    for i in file_data:
        if i.startswith("#"):
            continue
        line = i.split()
        if line[0] not in cup_dict:
            cup_dict[line[0]] = []
        cup_dict[line[0]].append(line[1:3])
    return cup_dict


def filter_cup(chrom, pos, cup_dict):
    cup_var = False
    if chrom not in cup_dict:
        cup_var = True
    else:
        for ele in cup_dict[chrom]:
            if float(pos) > float(ele[0]) and float(pos) <= float(ele[1]):
                cup_var = True
                break
    return cup_var


def extract_neg(vcf_file, out_file, pos_var, cup_dict, reg_dict):
    converter = get_lifter("hg19","hg38")
    pops = "AF","EAS_AF","AMR_AF","AFR_AF","EUR_AF","SAS_AF"
    total_neg = 0
    with open(vcf_file) as f:
        var_data = f.readlines()
    header_num = 0
    for i in var_data:
        if i.startswith("#"):
            header_num += 1
        else:
            break
    var_data = var_data[header_num:]
    random.seed(0)
    neg_vars = random.sample(var_data,1000000)
    with open(out_file, 'w') as output:
        for item in neg_vars:
            variant = item.split("\t")
            var_info = variant[-1].strip().split(";")
            #print(var_info)
            if ("MULTI_ALLELIC" in variant[-1] or "EX_TARGET" in variant[-1] or "VT=SNP" not in variant[-1]):
                continue
            info_dict = {j.split("=")[0]:j.split("=")[1] for j in var_info}
            pop_freq = [float(info_dict[pop_name]) for pop_name in pops]
            min_freq = min(pop_freq)
            if (len(variant[3]) == 1 and len(variant[4]) == 1 and min_freq > 0.05):
                #cup filter
                chrom, pos = variant[:2]
                chrom = "chr" + chrom
                cup_var = filter_cup(chrom, pos, cup_dict)
                if cup_var == True:
                    pass
                else:
                    chrom_num = chrom.replace("chr","")
                    new_coord = converter[chrom_num][int(pos)]
                    if new_coord == []:
                        pass
                    else:
                        new_chrom = new_coord[0][0]
                        new_pos = str(new_coord[0][1])
                        snp = new_chrom + ":" + new_pos
                        if snp not in pos_var:
                            reg_var = filter_cup(new_chrom, new_pos, reg_dict)
                            if reg_var == False:
                                total_neg += 1
                                print("Writing negative non-functional SNP num. ", total_neg)
                                output.write(new_chrom + "\t" + new_pos + "\t" + variant[3] + "\t" + variant[4] + "\n")
            else:
                pass
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Outputs negative non-functional SNPs to file')
    parser.add_argument('-v', '--vcf_file', default = "/media/hmg/InternalHDD/anil/neurally/ALL.wgs.phase3_shapeit2_mvncall_integrated_v5c.20130502.sites.vcf", help='1000 genome phase3 vcf file')
    parser.add_argument('-g', '--gwas_file', default = "/media/hmg/InternalHDD/anil/neurally/gwas_catalog_allvars.txt", help='gwas catalog all variants coordinate file')
    parser.add_argument('-e', '--eqtl_file', default = "/media/hmg/InternalHDD/anil/neurally/eqtl_allvars.txt", help='eQTL all variants coordinate file')
    parser.add_argument('-o', '--out_file', default="neg_vars.txt", help='output file having negative non-functional SNPs without overlapping gwas and eQTL variants')
    parser.add_argument('-b', '--exon_bed',default = "/home/hmg/Downloads/regulomevar/gencode_exons_modif_canonical.bed", help='bed file with canonical exon coordinates according to hg38 build')
    parser.add_argument('-c', '--ccre_bed', default = "/media/hmg/InternalHDD/anil/neurally/encode_ccre.bed", help='bed file with encode candidate cis-regulatory coordinates according to hg38 build')
    parser.add_argument('--cup_file', default="/home/hmg/Downloads/regulomevar/gwas/FASTA_BED.ALL_GRCh37.novel_CUPs.bed", help='bed file with conversion-unstable position coordinates, cup file for the respective build should be used')
    args = parser.parse_args()
    
    
    print("Extracting GWAS and eQTL variants to set...")
    pos_var = extract_pos(args.gwas_file, args.eqtl_file)
    
    print("Adding conversion-unstable positions to dictionary...")
    cup_dict = {}
    cup_dict = read_bed(args.cup_file, cup_dict)
    
    print("Adding coding and encode candidate cis-regulatory region coordinates to dictionary...")
    reg_dict = {}
    reg_dict = read_bed(args.exon_bed, reg_dict)
    reg_dict = read_bed(args.ccre_bed, reg_dict)
    
    print("Writing negative non-functional SNPs to file...")
    extract_neg(args.vcf_file, args.out_file, pos_var, cup_dict, reg_dict)

