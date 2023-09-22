import os, errno, re, argparse

def extract_vars(gwas_file, out_file):
    gwas_var = []
    with open(gwas_file, 'r') as f:
        gwas_data = f.readlines()
    gwas_data = gwas_data[1:]
    with open(out_file, 'w') as output:
        for i in gwas_data:
            line = i.split("\t")         
            chrom = line[11]
            pos = line[12]
            if (chrom == "" or pos == "" or len(chrom)>2):
                pass
            else:
                chrom = "chr" + chrom
                output.write(chrom + "\t" + pos + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Outputs all GWAS Catalog variant coordinates to file')
    parser.add_argument('-g', '--gwas_file', default = "/home/hmg/gwas_catalog_all_v1.0", help='gwas catalog all dataset')
    parser.add_argument('-o', '--out_file', default="gwas_catalog_vars.txt", help='output file having gwas catalog variant coordinates')
    args = parser.parse_args()

    #raises exception if the gwas catalog file does not exist in the current location
    
    if not os.path.isfile(args.gwas_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.gwas_file)

    print("Writing significant variant coordinates from gwas catalog dataset...")
    extract_vars(args.gwas_file, args.out_file)


