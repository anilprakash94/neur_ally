# Neur-Ally
**Neur-Ally** is a deep learning model for predicting the regulatory effect of neurologic single nucleotide variations

![alt text](https://github.com/anilprakash94/neur_ally/blob/main/datasets/neurally.png?raw=true)


## Dependencies

* Python version = 3.8.16
* OS = Ubuntu 20.04.4 

### Python Libraries

* tensorflow (2.7.0) (Tensorflow dependencies recommended for the specific version is required for GPU support)

* numpy (1.23.1)

* matplotlib (3.4.2)

* seaborn (0.11.1)

* pandas (1.3.1)

* pyfaidx (0.6.2)

* adjustText (0.8)

* liftover (1.1.16)

* scikit-learn (1.2.2)


### Required Files

* Human reference genome (hg38.fa)

* Encode Exclusion files ("GRCh38_unified_blacklist.bed","dukeExcludeRegions.bed")

* Encode epigenomic processed narrowpeak bed files for training input data generation

* GWAS Catalog dataset (gwas_catalog_all_v1.0)

* dbSNP common variant dataset (common_all_20180418.vcf)

* GTEx portal v8 signif_variant_gene_pairs.txt files.

* "gencode_exons_modif_canonical.bed" (bed file from UCSC with canonical exon coordinates according to hg38 build)

* "ALL.wgs.phase3_shapeit2_mvncall_integrated_v5c.20130502.sites.vcf" (1000 genome phase3 vcf file)

* "encode_ccre.bed" (bed file from UCSC with encode candidate cis-regulatory coordinates according to hg38 build)

* "FASTA_BED.ALL_GRCh37.novel_CUPs.bed" (bed file with conversion-unstable position coordinates, can be downloaded from "https://github.com/cathaloruaidh/genomeBuildConversion".)


## Scripts

**Neur-Ally** can be trained on epigenomic datasets and used for variant effect prediction.

### Scripts for preprocessing, training and testing

The scripts for running the model are:
```
dataprocess.py

--Creates training, validation and testing dataset from encode .bed files
```
```
neurally_train.py

--Trains the model on training data.
```
```
neurally_test.py

--Testing the model after training. The weights files for the 27th epoch are provided in the repository which can be directly used for testing the model.
```

### Scripts for variant effect prediction

The scripts for predicting the regulatory effect of non-coding variants:
```
gwasall.py and gwaspos.py

--gwasall.py outputs all GWAS Catalog variant coordinates to file, gwaspos.py outputs neurologic GWAS variant coordinates to file as the positive variant set
```
```
eqtlall.py and eqtlpos.py

--eqtlall.py outputs all eQTL variant coordinates to file, eqtlpos.py outputs top 1000 neurologic eQTL variant coordinates to file as the positive variant set
```
```
negset.py and negfile.py

-- negset.py outputs negative non-functional SNPs to file and negfile.py saves sad-score array of negative non-functional SNP set to .npy file
```
```
asdpos.py

--outputs Autism Spectrum Disroder(ASD) GWAS variant coordinates to file
```
```
sadscores.py

--Variant effect prediction using in-silico mutagenesis, creates output file with SAD scores.
```
```
evalue.py and signifeval.py

--evalue.py outputs e-value of positive SNPs and signifeval.py writes significant e-values of positive SNPs to file
```

### Scripts for plotting the results

```
heatmap_eval.py and manhattan_eval.py

--heatmap_eval.py plots heatmap of SNPs with significant e-values and manhattan_eval.py plots manhattan plot of SNPs with significant e-values
```

## Usage

### Running the model

```
git clone https://github.com/anilprakash94/neur_ally.git neur_ally

cd neur_ally

```
Then, run the programs according to the requirements and instructions listed in README.md.

For example:

python3 dataprocess.py -h

```
usage: dataprocess.py [-h] [-d DATASET_PATH] [-e FILE_LIST]
                      [--excl_files EXCL_FILES] [-i INPUT_BINS]

Neur-Ally Data Preprocessing

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET_PATH, --dataset_path DATASET_PATH
                        path of folder containing encode datasets
  -e FILE_LIST, --file_list FILE_LIST
                        specify the file to which encode filenames are written
  --excl_files EXCL_FILES
                        path of bed files having exclusion regions
  -i INPUT_BINS, --input_bins INPUT_BINS
                        specify the file name to which preprocessed input data
                        is written

```

