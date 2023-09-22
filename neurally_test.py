import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
import errno
import numpy as np
import math
import argparse
from pyfaidx import Fasta
import matplotlib.pyplot as plt
import re
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

from neurally_train import *


def model_testing(test_dataset, args, encfiles, test_model, ref_genome):
    roc_auc = []
    baseline_pr = []
    pr_auc = []
    test_steps = len(test_dataset) // args.batch_size
    label_num = len(encfiles)
    #metric calculation is carried out on a subset of targets because the entire testing samples and labels won't fit in memory..
    #test_bin determines the subset target used for testing
    test_bin = (label_num * 10) // 100
    for cls in range(0, label_num, test_bin):
        for sample in range(test_steps):
            print(f'Running test samples:{str(sample*args.batch_size) + ":" + str((sample+1)*args.batch_size)}')
            test_batch = test_dataset[sample*args.batch_size:(sample+1)*args.batch_size]
            x_test, y_test = process_element(test_batch, ref_genome, args)
            y_pred = test_model(input_data=x_test)
            if (cls + test_bin) > label_num:
                y_test = y_test[:,cls:label_num]
                y_pred = y_pred[:,cls:label_num]
            else:        
                y_test = y_test[:,cls:cls+test_bin]
                y_pred = y_pred[:,cls:cls+test_bin]
            if sample == 0:
                test_true = y_test.flatten().tolist()
                test_pred = y_pred.numpy().flatten().tolist()
            else:
                test_true.extend(y_test.flatten().tolist())
                test_pred.extend(y_pred.numpy().flatten().tolist())
        num_class = y_test.shape[-1]
        total_peaks = len(test_true) // num_class
        test_true = np.array(test_true).reshape(total_peaks,num_class)
        test_pred = np.array(test_pred).reshape(total_peaks,num_class)
        for num in range(num_class):
            true_labels = test_true[:,num]
            pred_labels = test_pred[:,num]
            roc_auc.append(roc_auc_score(true_labels,pred_labels))
            baseline_pr.append(np.mean(true_labels))
            pre, rec, _ = precision_recall_curve(true_labels, pred_labels)
            sort_pre = np.argsort(pre)
            pre = np.array(pre)[sort_pre]
            rec = np.array(rec)[sort_pre]
            pr_auc.append(auc(pre, rec))
    out_path = args.model_dir+"/"+args.model_name+"/"+args.out_file
    with open(out_path, "w") as tsv_file:
        tsv_file.write("Sample" + "\t" + "Assay" + "\t" + "Bed_file" + "\t" + "AUROC" + "\t" + "PR-AUC" + "\t" +  "Baseline_PR-AUC" + "\n")
        for cls in range(label_num):
            sample_type, assay, bed_file = encfiles[cls].split("_")[0:3]
            tsv_file.write(sample_type + "\t" + assay + "\t" + bed_file + "\t" + str(roc_auc[cls]) + "\t" + str(pr_auc[cls]) + "\t" + str(baseline_pr[cls]) + "\n")
        tsv_file.write("" + "\t" + "" + "\t" + "Mean_values" + "\t" + str(np.mean(roc_auc)) + "\t" + str(np.mean(pr_auc)) + "\t" + str(np.mean(baseline_pr)) + "\n")   



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neur-Ally Testing using sklearn metrics')
    parser.add_argument("--epoch_num", type=int, help="epoch number of weights file having minimum validation loss")
    parser.add_argument('--dim', type=int, default=128, help='dimensions of input after embedding')
    parser.add_argument('--batch_size', type=int, default=160, help='specify the batch_size needed for training')
    parser.add_argument('--seq_len', type=int, default=2000, help='total flanking+bin sequence length for each input')
    parser.add_argument('--heads', type=int, default=4, help='number of multi head attention heads')
    parser.add_argument('-d', '--dataset_files', dest = "dataset_files" , default="datasets/encfiles.txt", help='text file containing names of encode dataset files')
    parser.add_argument('--model_dir', dest = "model_dir", default = "Models", help='specify the output folder name for saving the model')
    parser.add_argument('-r', '--ref_fasta', dest = "ref_fasta", default = "datasets/hg38.fa", help='specify the reference genome fasta file')
    parser.add_argument('--test_file', default="datasets/input_bins_test.txt", help='testing data file')
    parser.add_argument('--out_file', default="out_file.txt", help='name of output file having testing results')
    parser.add_argument('--model_name', default="neurally", help='specify the name of the model under study')
    args, unknown = parser.parse_known_args()
    
    
    #raises exception if the input testing file does not exist in the current location

    if not os.path.isfile(args.test_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.test_file)
    
    #raises exception if the reference genome fasta file does not exist in the current location
    
    if not os.path.isfile(args.ref_fasta):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.ref_fasta)
        
    print("Saving reference genome to variable...")
    ref_genome = Fasta(args.ref_fasta)
    
    print("Extracting filenames...")
    encfiles = extract_names(args)
    label_num = len(encfiles)
    
    #create positional encoding array
    pos = pos_encode(args.seq_len, args.dim)
    print("Initializing the model...")
    test_model = Modelsubclass(args, label_num, pos)
    
    print("Reading testing dataset from file...")
    with open(args.test_file) as f:
        test_dataset = f.readlines()
    
    #create one testing batch for running model once
    x_single, _ = one_train_batch(args, test_dataset, ref_genome)
    
    test_model(input_data=x_single)
    
    print("Loading weights from file...")
    model_dir = args.model_dir
    dir_path = weights_filepath(model_dir+"/"+args.model_name)
    file_names = os.listdir(dir_path)
    idx_files = []
    for names in file_names:
        if names.endswith(".index"):
            idx_files.append(names)
    
    for idx in idx_files:
        if int(re.search('weights.(.+?)-', idx).group(1)) == args.epoch_num:
            weights_file = idx[0:-6]
    
    weights_path = dir_path + "/" + weights_file
    status = test_model.load_weights(weights_path).expect_partial() 
    
    print("Asserting matching of weights...")
    status.assert_existing_objects_matched()
    
    print("Testing model...")
    model_testing(test_dataset, args, encfiles, test_model, ref_genome)
    
