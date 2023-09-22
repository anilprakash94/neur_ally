import os
import errno
import argparse


#loading all encode peak files in the folder

def extract_files(dataset_path):
    encfiles = []
    file_names = os.listdir(dataset_path)
    for names in file_names:
        if names.endswith(".bed"):
            encfiles.append(names)
    return encfiles


def write_filenames(args, encfiles):
    with open(args.file_list, "w") as output:
        for name in encfiles:
            output.write(name + "\n")


def concat_bins(dataset_path, encfiles):
    enc_dict = {}
    for path in encfiles:
        filepath = dataset_path + "/" + path
        with open(filepath,'r') as f:
            file = f.readlines()
        for i in range(len(file)):
            chrom, start, end = file[i].split("\t")[0:3]
            start, end  = int(start), int(end) 
            for bins in range( 200 * (start // 200), 200 * ( (end // 200) + 1 ), 200 ):
                encode_bins = [chrom, bins, bins + 200, path ]
                if encode_bins[2] - start > 100 and end - encode_bins[1] > 100 and end - start > 100 :
                    enc_key = encode_bins[0] + ":" + str(encode_bins[1]) + ":" + str(encode_bins[2])
                    if enc_key not in enc_dict:
                        enc_dict[enc_key] = []
                    enc_dict[enc_key].append(encode_bins[3])  
    return enc_dict


#function for converting bed into dict object

def bed_todict(bed_file,exclude_dict):
    with open(bed_file, 'r') as f:
        file = f.readlines()
    for i in file:
        line = i.strip().split()
        if line[0] not in exclude_dict:
            exclude_dict[line[0]] = []
        exclude_dict[line[0]].append(line[1] + ":" + line[2])
    return exclude_dict

#remove blacklisted and low mappability regions from input regions

def exclude_regions(enc_dict, exclude_dict):
    input_dict = {}
    for x,y in enc_dict.items():
        key_item = x.split(":")
        chrom, start, end = key_item
        if chrom in exclude_dict:
            for z in exclude_dict[chrom]:
                excl = z.split(":")
                if int(end) > int(excl[0]) and int(start) < int(excl[1]):
                    break
            else:
                input_dict[x] = y
        else:
            input_dict[x] = y
    return input_dict


def create_input(input_dict, args, train_chr):
    input_bins = args.input_bins
    with open(input_bins+"_train.txt", "w") as output:
        for key,value in input_dict.items():
            ele = key.split(":")
            if ele[0] in train_chr:
                labels = ""
                for i in encfiles:
                    if i in input_dict[key]:
                        labels += str(1)
                    else:
                        labels += str(0)
                output.write(ele[0] + "\t" + ele[1] + "\t" + ele[2] + "\t" + labels + "\n")
            else:
                pass
    with open(input_bins+"_val.txt", "w") as output:
        for key,value in input_dict.items():
            ele = key.split(":")
            if ele[0] == "chr7":
                labels = ""
                for i in encfiles:
                    if i in input_dict[key]:
                        labels += str(1)
                    else:
                        labels += str(0)
                output.write(ele[0] + "\t" + ele[1] + "\t" + ele[2] + "\t" + labels + "\n")
            else:
                pass
    with open(input_bins+"_test.txt", "w") as output:
        for key,value in input_dict.items():
            ele = key.split(":")
            if ele[0] == "chr8":
                labels = ""
                for i in encfiles:
                    if i in input_dict[key]:
                        labels += str(1)
                    else:
                        labels += str(0)
                output.write(ele[0] + "\t" + ele[1] + "\t" + ele[2] + "\t" + labels + "\n")
            else:
                pass




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neur-Ally Data Preprocessing')
    parser.add_argument('-d', '--dataset_path', dest = "dataset_path" , default="home/hmg/raw_data", help='path of folder containing encode datasets')
    parser.add_argument('-e', '--file_list', dest = "file_list", default = "datasets/encfiles.txt", help='specify the file to which encode filenames are written')
    parser.add_argument('--excl_files', default = ["datasets/GRCh38_unified_blacklist.bed","datasets/dukeExcludeRegions.bed"], help='path of bed files having exclusion regions')
    parser.add_argument('-i', '--input_bins', dest = "input_bins", default = "datasets/input_bins", help='specify the file name to which preprocessed input data is written')

    args = parser.parse_args()

    dataset_path = args.dataset_path

    #raises exception if the dataset directory path does not exist

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_path)

    print("Extracting all encode filenames...")
    encfiles = extract_files(dataset_path)

    print("Saving list of encode filenames into text file...")
    write_filenames(args, encfiles)

    print("Saving encode peaks into dict...")
    enc_dict = concat_bins(dataset_path, encfiles)
        
    #load bed file of excluded regions
    exclude_dict = {}
    for bed_path in args.excl_files:
        exclude_dict = bed_todict(bed_path,exclude_dict)
    
    print("Removing encode peaks from blacklisted and low mappability regions...")
    input_dict = exclude_regions(enc_dict, exclude_dict)
    
    train_chr = ["chr1","chr2","chr3","chr4","chr5","chr6","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22","chrX","chrY"]
    print("Writing data of all bins to file...")
    create_input(input_dict, args, train_chr)


