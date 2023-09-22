import os, re, argparse


#loading all significant variant-gene pair files in the folder

def extract_files(dataset_path):
    gtex_files = []
    file_names = os.listdir(dataset_path)
    for names in file_names:
        if names.endswith(".txt"):
            gtex_files.append(names)
    return gtex_files


def extract_vars(dataset_path, gtex_files):
    eqtl_set = set()
    for path in gtex_files:
        if dataset_path.endswith("/"):
            filepath = dataset_path + path
        else:
            filepath = dataset_path + "/" + path
        with open(filepath,'r') as f:
            txt_file = f.readlines()
        txt_file = txt_file[1:]
        for i in txt_file:
            variant = i.split("\t")[0]
            chrom, pos, ref, alt, build = variant.split("_")
            if (len(ref) == 1 and len(alt) == 1):
                eqtl_set.add(chrom+":"+pos+":"+ref+":"+alt) 
    return eqtl_set


def save_vars(eqtl_set, out_file):
    with open(out_file, 'w') as output:
        for i in eqtl_set:
            line = i.split(":")         
            output.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + line[3] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Outputs all eQTL variant coordinates to file')
    parser.add_argument('-d', '--dataset_path', dest = "dataset_path" , default="/home/hmg/eqtl_data", help='path of folder containing eqtl significant variant-gene pair files in .txt format')
    parser.add_argument('-o', '--out_file', default="eqtl_allvars.txt, help='output file having eQTL variant coordinates')
    args = parser.parse_args()
    
    print("Extracting all filenames with significant variant-gene pairs...")
    gtex_files = extract_files(args.dataset_path)
    
    print("Adding significant eQTL coordinates to a set...")
    eqtl_set = extract_vars(args.dataset_path, gtex_files)
    
    print("Writing eQTL variant coordinates to file...")
    save_vars(eqtl_set, args.out_file)


