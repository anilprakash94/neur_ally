import tensorflow as tf
import tensorflow.keras.backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

import os
import errno
import numpy as np
import math
import argparse
import pickle
import re
import random
import math
from pyfaidx import Fasta


def extract_names(args):
    dataset_files = args.dataset_files
    encfiles = []
    with open(dataset_files, 'r') as f:
        file = f.readlines()
    for ele in file:
        encfiles.append(ele.strip())
    return encfiles
        

def pos_to_seq(data, seq_len, ref_genome, peak_length = 200):
    #peak chromosome coordinates are " 0-based exclusive"
    left_len = int(((seq_len-peak_length)/2)) 
    right_len = int(((seq_len-peak_length)/2))
    start = max(0,int(data[1]) - left_len)
    end = int(data[2]) + right_len
    left_seq = ref_genome[data[0]] [start : int(data[1])].seq
    if len(left_seq) < left_len:
        left_seq = 'N' * (left_len - len(left_seq) )  + left_seq 
    right_seq = ref_genome[data[0]] [int(data[2]) : end].seq
    if len(right_seq) < right_len:
        right_seq += 'N' * (right_len - len(right_seq) ) 
    peak_seq = ref_genome[data[0]] [int(data[1]) : int(data[2]) ].seq
    seq_data = left_seq + peak_seq + right_seq
    return seq_data


def vectorization(seq_data):
    nuc = 'nagct'
    for index, allele in enumerate(nuc):
        seq_data = seq_data.replace(allele, str(index))
        seq_data = seq_data.replace(allele.upper(), str(index))
    input_array = np.fromiter(list(seq_data), dtype=np.int8)
    return input_array


def create_mask(input_data, dim):
    mask = (input_data == 0)
    mask_array = tf.cast(mask, tf.float32)
    mask_array = tf.tile(tf.expand_dims(mask_array, axis=-1), [1,1,dim])
    mask_array = tf.constant([1], dtype= tf.float32) - mask_array
    return mask_array


def pos_encode(seq_len, dim):
    pos = np.tile(np.expand_dims(np.arange(seq_len, dtype=np.float32), 1), (1,dim))
    i = np.arange(dim, dtype=np.float32)
    for x,y in enumerate(pos):
        pos[x] = y / (10000 ** ((2 * (i//2)) / dim ) )
    pos[:, 0::2] = np.sin(pos[:, 0::2])
    pos[:, 1::2] = np.cos(pos[:, 1::2])    
    return pos


def pos_encode_batch(input_emb, mask_array, pos):
    pos = tf.tile(tf.expand_dims(pos, axis=0), [K.eval(K.shape(input_emb)[0]),1,1])
    pos = pos * mask_array
    input_pos = input_emb + pos
    return input_pos


def process_element(batch_train, ref_genome, args):
    y_train = []
    seq_data = ""
    for i in batch_train:
        data = i.strip().split()
        seq_data += pos_to_seq(data[0:3], args.seq_len, ref_genome)
        output_array = [int(y) for y in data[3]]
        y_train.append(np.array(output_array, ndmin=2))
    input_array = vectorization(seq_data)
    x_train = np.reshape(input_array, (args.batch_size,args.seq_len))
    y_train = np.concatenate(y_train, axis=0)
    return x_train, y_train


class Custom_Generator(tf.keras.utils.Sequence):
    def __init__(self,train_data, args, ref_genome):
        self.args = args
        self.train_data = train_data
        self.train_indices = list(range(len(train_data)))
        self.batch_size = args.batch_size
        self.ref_genome = ref_genome
        self.on_epoch_end()
        self.iterate = 0
    
    def __len__(self):
        return len(self.train_indices) // self.batch_size
    
    def __getitem__(self, idx):
        index_list = self.train_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_train = []
        for index in index_list:
            batch_train.append(self.train_data[index])
        x_train, y_train = process_element(batch_train, self.ref_genome, self.args)
        return x_train, y_train
    
    def on_epoch_end(self):
        np.random.shuffle(self.train_indices)
    
    def __next__(self):
        if self.iterate >= self.__len__():
            self.iterate = 0
        self.iterate += 1
        return self.__getitem__(self.iterate-1)


def weights_filepath(model_dir):
    current_dir = os.getcwd()
    dir_path = os.path.join(current_dir,model_dir)
    if os.path.isdir(dir_path):
        pass   
    else:
        os.makedirs(dir_path)
    return dir_path


class emb2pos(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def call(self, inp_tensors):
        dim = self.args.dim
        seq_len = self.args.seq_len
        input_emb = inp_tensors[0]
        input_data = inp_tensors[1]
        pos_array = inp_tensors[2]
        mask_array = create_mask(input_data, dim)
        input_emb = input_emb * mask_array
        input_pos = pos_encode_batch(input_emb, mask_array, pos_array)
        return input_pos


class MultiHead_attn(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.w1 = self.add_weight(name='w1', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.w2 = self.add_weight(name='w2', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.w3 = self.add_weight(name='w3', shape=(self.dim, self.dim), initializer="glorot_uniform", trainable=True)
        self.b1 = self.add_weight(name='b1', shape=(self.dim,), initializer="zeros", trainable=True)
        self.b2 = self.add_weight(name='b2', shape=(self.dim,), initializer="zeros", trainable=True)
        self.b3 = self.add_weight(name='b3', shape=(self.dim,), initializer="zeros", trainable=True)
    
    def call(self, inputs, training= None):
        dim = self.dim
        heads = self.args.heads
        Query = tf.matmul(inputs, self.w1) + self.b1
        Key = tf.matmul(inputs, self.w2) + self.b2
        Value = tf.matmul(inputs, self.w3) + self.b3
        Query = tf.concat(tf.split(Query, heads, axis=-1), axis=0)
        Key = tf.concat(tf.split(Key, heads, axis=-1), axis=0)
        Value = tf.concat(tf.split(Value, heads, axis=-1), axis=0)   
        out = (tf.matmul(Query, tf.transpose(Key, [0, 2, 1]))) / math.sqrt(K.int_shape(Key)[-1])
        out = tf.nn.softmax(out)
        if training:
            out = tf.nn.experimental.stateless_dropout(out, rate=0.2, seed=[1, 0])
        out = tf.matmul(out, Value)
        out = tf.concat(tf.split(out, heads, axis=0), axis=-1)
        out = out + inputs
        out = tf.keras.layers.LayerNormalization(axis=-1)(out)
        return out


class squeeze_tensor(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    
    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.sw1 =  self.add_weight(name='sw1', shape=(self.dim, 1), initializer="he_uniform", trainable=True)
        self.sb1 = self.add_weight(name='sb1', shape=(1,), initializer="zeros", trainable=True)
    
    def call(self, output):
        output = tf.matmul(output, self.sw1)
        output = tf.nn.leaky_relu(output + self.sb1)
        output = tf.squeeze(output, axis= -1)  
        output = tf.keras.layers.LayerNormalization(axis=-1)(output)
        return output


class opt_weights(tf.keras.callbacks.Callback):
    def __init__(self, dir_path, opt): 
        super().__init__()
        self.dir_path = dir_path
        self.opt = opt
    
    def on_epoch_end(self, epoch, logs={}):
        opt_weights = tf.keras.optimizers.Adam.get_weights(self.opt)
        with open(self.dir_path+'/optimizer.pkl', 'wb') as f:
            pickle.dump(opt_weights, f)



class Modelsubclass(tf.keras.Model):
    def __init__(self, args, label_num, pos_array):
        super().__init__()
        self.pos_array = pos_array
        self.embed1 = tf.keras.layers.Embedding(5, args.dim)
        self.pos = emb2pos(args)
        self.conv1 = tf.keras.layers.Conv1D(64, 10, activation='relu')
        self.max1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=3, padding='valid') 
        self.drop1 = tf.keras.layers.Dropout(0.2)
        self.conv2 = tf.keras.layers.Conv1D(32, 10, activation='relu')
        self.max2 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=3, padding='valid') 
        self.drop2 = tf.keras.layers.Dropout(0.2)
        self.att1 = MultiHead_attn(args)
        self.att2 = MultiHead_attn(args)
        self.squeeze = squeeze_tensor()
        self.dense3 = tf.keras.layers.Dense(label_num, activation='sigmoid')
    
    def call(self, input_data, training=False):
        input_emb = self.embed1(input_data)
        pos_output =  self.pos([input_emb, input_data, self.pos_array])
        inputs = self.conv1(pos_output)
        inputs = self.max1(inputs)
        inputs = self.drop1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.max2(inputs)
        inputs = self.drop2(inputs)
        out = self.att1(inputs, training = training)
        output = self.att2(out, training = training)
        output = self.squeeze(output)
        output =self.dense3(output)
        return output


def one_train_batch(args, train_data, ref_genome):
    batch_train = []
    index_list = list(range(args.batch_size))
    for index in index_list:
        batch_train.append(train_data[index])
    x_single, y_single = process_element(batch_train, ref_genome, args)
    return x_single, y_single


@tf.autograph.experimental.do_not_convert
def custom_loss(y_train, y_pred):
    y_train = tf.cast(y_train, tf.float32)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    pos = y_train * K.log(y_pred)
    neg = (1 - y_train) * K.log(1 - y_pred)
    loss = -K.mean(pos + neg, axis = 1)
    return loss


def model_run(train_data, val_data, args, ref_genome, label_num, pos_array):
    print("Instantiating model...")
    model = Modelsubclass(args, label_num, pos_array)
    opt=tf.keras.optimizers.Adam(learning_rate=args.lr)
    dir_path = weights_filepath(args.model_dir+"/"+args.model_name)
    
    save_opt = opt_weights(dir_path, opt)
    
    roc_auc = tf.keras.metrics.AUC()
    pr_auc = tf.keras.metrics.AUC(curve='PR')
        
    print("Compiling model...")
    model.compile(loss=custom_loss, optimizer=opt, metrics=[roc_auc, pr_auc], run_eagerly=True)
    
    train_generator = Custom_Generator(train_data, args, ref_genome)

    val_generator = Custom_Generator(val_data, args, ref_genome)
    
    resume_train = args.resume_train
    log_name = dir_path + "/" + args.model_name + "_train.log"
    
    if resume_train == False:
        csv_logger = tf.keras.callbacks.CSVLogger(log_name)
    else:
        csv_logger = tf.keras.callbacks.CSVLogger(log_name, append=True)
    
    weights_path = os.path.join(dir_path, "weights.{epoch:02d}-{val_loss:.2f}")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=False)

    
    if resume_train == False:
        print("Training new model...")
        model.fit(x=train_generator, epochs=500, validation_data=val_generator, callbacks=[model_checkpoint_callback, save_opt, csv_logger],verbose= 1)
    else:
        checkpoint_file = dir_path + "/checkpoint"
        with open(checkpoint_file, 'r') as f:
            file = f.readlines()
        
        weights_file = re.search('path: "(.+?)"', file[0]).group(1)
        epoch_num = int(re.search('weights.(.+?)-', weights_file).group(1))
        weights_file = os.path.join(dir_path, weights_file)
        
        print("Loading weights from previously trained model...")
        status = model.load_weights(weights_file).expect_partial()
        status.assert_existing_objects_matched()
        
        print("Training single input batch...")
        x_single, y_single = one_train_batch(args, train_data, ref_genome)
        model.fit(x_single, y_single)
        
        with open(dir_path+'/optimizer.pkl', 'rb') as f:
            weight_variable = pickle.load(f)
        
        print("Loading optimizer weights from file...")
        model.optimizer.set_weights(weight_variable)
        
        print("Resuming training of model from last saved epoch...")
        model.fit(x=train_generator, epochs=500, validation_data=val_generator, callbacks=[model_checkpoint_callback, save_opt, csv_logger], initial_epoch=epoch_num, verbose= 1)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neur-Ally Training')
    parser.add_argument('--dim', type=int, default=128, help='dimensions of input after embedding')
    parser.add_argument('--batch_size', type=int, default=100, help='specify the batch_size needed for training')
    parser.add_argument('--seq_len', type=int, default=2000, help='total flanking+bin sequence length for each input')
    parser.add_argument('--heads', type=int, default=4, help='number of multi head attention heads')
    parser.add_argument('--resume_train', dest = "resume_train", action='store_true', help='default to False when the command-line argument is not present, if true already trained weights are loaded to resume training')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate float value')
    parser.add_argument('--model_dir', dest = "model_dir", default = "Models", help='specify the output folder name for saving the model')
    parser.add_argument('-r', '--ref_fasta', dest = "ref_fasta", default = "datasets/hg38.fa", help='specify the reference genome fasta file')
    parser.add_argument('--train_file', default="datasets/input_bins_train.txt", help='training data file')
    parser.add_argument('--val_file', default="datasets/input_bins_val.txt", help='validation data file')
    parser.add_argument('-d', '--dataset_files', dest = "dataset_files" , default="datasets/encfiles.txt", help='text file containing names of encode dataset files')
    parser.add_argument('--model_name', default="neurally", help='specify the name of the model under study')
    args, unknown = parser.parse_known_args()
    
    
    #raises exception if the input training file does not exist in the current location

    if not os.path.isfile(args.train_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.train_file)
    
    #raises exception if the input validation file does not exist in the current location

    if not os.path.isfile(args.val_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.val_file)
    
    #raises exception if the reference genome fasta file does not exist in the current location
    
    if not os.path.isfile(args.ref_fasta):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.ref_fasta)
    
    print("Extracting filenames...")
    encfiles = extract_names(args)
    label_num = len(encfiles)
    print("Saving reference genome to variable...")
    ref_genome = Fasta(args.ref_fasta)
    
    print("Reading training and validation datasets from file...")
    with open(args.train_file) as f:
        train_data = f.readlines()
    with open(args.val_file) as f:
        val_data = f.readlines()
    #create positional encoding array
    pos_array = pos_encode(args.seq_len, args.dim)
    #model run function
    model_run(train_data, val_data, args, ref_genome, label_num, pos_array)     
    
