# encoding: utf-8
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=10000)
from ExtractData import Dataset_aux,Dataset_tar
from GetTest import get_test_list
from time import time
import math, os


def ini_word_embed_aux(num_words_aux, latent_dim):
    word_embeds_aux = np.random.rand(num_words_aux, latent_dim)
    return word_embeds_aux
def ini_word_embed_tar(num_words_tar, latent_dim):
    word_embeds_tar = np.random.rand(num_words_tar, latent_dim)
    return word_embeds_tar

def word2vec_word_embed(num_words, latent_dim, path, word_id_dict):
    word2vect_embed_mtrx = np.zeros((num_words, latent_dim))
    with open(path, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            row_id = word_id_dict.get(arr[0])
            vect = arr[1].strip().split(" ")
            for i in range(len(vect)):
                word2vect_embed_mtrx[row_id, i] = float(vect[i])
            line = f.readline()

    return word2vect_embed_mtrx

def get_train_instance_aux(train_aux):
    user_input_aux, item_input_aux, rates_aux = [], [], []

    for (u, i) in train_aux.keys():
        # positive instance
        user_input_aux.append(u)
        item_input_aux.append(i)
        rates_aux.append(train_aux[u,i])
    return user_input_aux, item_input_aux, rates_aux

def get_train_instance_tar(train_tar):
    user_input_tar, item_input_tar, rates_tar = [], [], []

    for (u, i) in train_tar.keys():
        # positive instance
        user_input_tar.append(u)
        item_input_tar.append(i)
        rates_tar.append(train_tar[u,i])
    return user_input_tar, item_input_tar, rates_tar

def get_train_instance_batch_change_aux(count, batch_size_aux, user_input_aux, item_input_aux, ratings_aux, user_reviews_aux, item_reviews_aux):
    users_batch_aux, items_batch_aux, user_input_batch_aux, item_input_batch_aux, labels_batch_aux = [], [], [], [], []

    for idx in range(batch_size_aux):
        index = (count*batch_size_aux + idx) % len(user_input_aux)
        users_batch_aux.append(user_input_aux[index])
        items_batch_aux.append(item_input_aux[index])
        user_input_batch_aux.append(user_reviews_aux.get(user_input_aux[index]))
        item_input_batch_aux.append(item_reviews_aux.get(item_input_aux[index]))
        labels_batch_aux.append([ratings_aux[index]])

    return users_batch_aux, items_batch_aux, user_input_batch_aux, item_input_batch_aux, labels_batch_aux

def get_train_instance_batch_change_tar(count, batch_size_tar, user_input, item_input, ratings, user_reviews, item_reviews):
    users_batch_tar, items_batch_tar, user_input_batch_tar, item_input_batch_tar, labels_batch_tar = [], [], [], [], []

    for idx in range(batch_size_tar):
        index = (count*batch_size_tar + idx) % len(user_input_tar)
        users_batch_tar.append(user_input_tar[index])
        items_batch_tar.append(item_input_tar[index])
        user_input_batch_tar.append(user_reviews_tar.get(user_input_tar[index]))
        item_input_batch_tar.append(item_reviews_tar.get(item_input_tar[index]))
        labels_batch_tar.append([ratings_tar[index]])

    return users_batch_tar, items_batch_tar, user_input_batch_tar, item_input_batch_tar, labels_batch_tar




def user_embeds_model(word_latent_factor, latent_dim, num_filters, windows_size, users_inputs_aux,users_inputs_tar, word_embeddings_aux, word_embeddings_tar,dropout_rate):
    user_reviews_representation_aux = tf.nn.embedding_lookup(word_embeddings_aux, users_inputs_aux)
    user_reviews_representation_tar = tf.nn.embedding_lookup(word_embeddings_tar, users_inputs_tar)
    user_reviews_representation_expnd_aux = tf.expand_dims(user_reviews_representation_aux, -1)
    user_reviews_representation_expnd_tar = tf.expand_dims(user_reviews_representation_tar, -1)

    #CNN layers
    W = tf.Variable(tf.truncated_normal([windows_size, word_latent_factor, 1, num_filters],stddev=0.3), name="user_W")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="user_b")
    
    conv_aux = tf.nn.conv2d(user_reviews_representation_expnd_aux, W, strides=[1,1,1,1], padding="VALID", name="user_conv_aux")
    
    h_aux = tf.nn.relu(tf.nn.bias_add(conv_aux, b), name="user_relu_aux")
    
    sec_dim_aux = h_aux.get_shape()[1]
    o_aux = tf.nn.avg_pool(
        h_aux,
        ksize=[1, sec_dim_aux, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="user_pool_aux")
    
    o_aux = tf.squeeze(o_aux)
    
    W1= tf.Variable(tf.truncated_normal([num_filters, latent_dim],stddev=0.3), name="user_W1")
    W1 = tf.nn.dropout(W1, dropout_rate)
    b1 = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="user_b1")
    user_vector_aux = tf.nn.relu_layer(o_aux, W1, b1, name="user_layer1")
    
    conv = tf.nn.conv2d(user_reviews_representation_expnd_tar, W, strides=[1,1,1,1], padding="VALID", name="user_conv")
    #print "conv", conv.get_shape()
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="user_relu")
    #print "h", h.get_shape()
    sec_dim = h.get_shape()[1]
    o = tf.nn.avg_pool(
        h,
        ksize=[1, sec_dim, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="user_pool")
    #print "o", o.get_shape()
    o = tf.squeeze(o)
    #print "o", o.get_shape()
    
    user_vector_tar = tf.nn.relu_layer(o, W1, b1, name="user_layer1_tar")


    return user_vector_aux,user_vector_tar


def item_embeds_model(word_latent_factor, latent_dim, num_filters, windows_size, items_inputs_aux, items_inputs_tar,word_embeddings_aux,word_embeddings_tar, dropout_rate):
    item_reviews_representation_aux = tf.nn.embedding_lookup(word_embeddings_aux, items_inputs_aux)
    item_reviews_representation_expnd_aux = tf.expand_dims(item_reviews_representation_aux, -1)
    item_reviews_representation_tar = tf.nn.embedding_lookup(word_embeddings_tar, items_inputs_tar)
    item_reviews_representation_expnd_tar = tf.expand_dims(item_reviews_representation_tar, -1)

    #CNN layers
    W = tf.Variable(tf.truncated_normal([windows_size, word_latent_factor, 1, num_filters],stddev=0.3), name="item_W")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="user_b")

    conv_aux = tf.nn.conv2d(item_reviews_representation_expnd_aux, W, strides=[1,1,1,1], padding="VALID", name="item_conv_aux")
    #print "conv", conv.get_shape()
    h_aux = tf.nn.relu(tf.nn.bias_add(conv_aux, b), name="item_relu_aux")
    #print "h", h.get_shape()
    sec_dim_aux = h_aux.get_shape()[1]
    o_aux = tf.nn.avg_pool(
        h_aux,
        ksize=[1, sec_dim_aux, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="item_pool_aux")
    #print "o", o.get_shape()
    o_aux = tf.squeeze(o_aux)
    #print "o", o.get_shape()
    W1 = tf.Variable(tf.truncated_normal([num_filters, latent_dim],stddev=0.3), name="item_W1")
    W1 = tf.nn.dropout(W1, dropout_rate)
    b1 = tf.Variable(tf.constant(0.1, shape=[latent_dim]), name="item_b")
    item_vector_aux = tf.nn.relu_layer(o_aux, W1, b1, name="item_layer1")
    
    conv = tf.nn.conv2d(item_reviews_representation_expnd_tar, W, strides=[1,1,1,1], padding="VALID", name="item_conv")
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="item_relu")
    sec_dim = h.get_shape()[1]
    o = tf.nn.avg_pool(
        h,
        ksize=[1, sec_dim, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="item_pool")
    o = tf.squeeze(o)
    item_vector_tar = tf.nn.relu_layer(o, W1, b1, name="item_layer1")
    return item_vector_aux,item_vector_tar

def AutoRec_model_user(hidden_neuron,latent_dim, num_user_tar,num_user_aux,user_vector_aux,user_vector_tar):
        
    units=100
    noise=0
    reg_lambda = 0.0
    
    stddev = 0.2
       # user_vector_tar_size = int(input.shape[1])
    
    corrupt_aux = tf.layers.dropout(user_vector_aux,rate= noise,training=True)
    corrupt_tar = tf.layers.dropout(user_vector_tar,rate= noise,training=True)

    ew = tf.get_variable(name="enc_weights_user",initializer=tf.truncated_normal(shape=[latent_dim, units],mean=0.0,stddev=stddev))
                                 
    eb = tf.get_variable(name="enc_biases_user",shape=[units],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
                                
            
    fc1_aux = tf.add(tf.matmul(corrupt_aux,ew),eb)
    Encoder_aux_user = tf.nn.leaky_relu(fc1_aux)               #leaky relu
    fc1_tar = tf.add(tf.matmul(corrupt_tar,ew),eb)
    Encoder_tar_user = tf.nn.leaky_relu(fc1_tar)

    dw = tf.transpose(ew)
    db = tf.get_variable(name="dec_biases_user",shape=[latent_dim],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
                                #initializer=tf.constant_initializer(0.0),dtype=tf.float32,
                               # regularizer=tf.contrib.layers.l2_regularizer(reg_lambda))    
            
    fc_aux = tf.add(tf.matmul(Encoder_aux_user,dw),db)
    Decoder_aux_user = tf.nn.leaky_relu(fc_aux)
    fc_tar = tf.add(tf.matmul(Encoder_tar_user,dw),db)
    Decoder_tar_user = tf.nn.leaky_relu(fc_tar)

      
    return Decoder_aux_user, Decoder_tar_user, Encoder_aux_user, Encoder_tar_user

def AutoRec_model_item(hidden_neuron,latent_dim, num_item_tar, num_item_aux, item_vector_aux,item_vector_tar):  
        
    units=100
    noise=0
    reg_lambda = 0.0
    
    stddev = 0.2    
           
    corrupt_aux = tf.layers.dropout(item_vector_aux,rate= noise,training=True)
    corrupt_tar = tf.layers.dropout(item_vector_tar,rate= noise,training=True)

    ew = tf.get_variable(name="enc_weights",initializer=tf.truncated_normal(shape=[latent_dim, units],mean=0.0,stddev=stddev),regularizer=tf.contrib.layers.l2_regularizer(reg_lambda))
                                 

    eb = tf.get_variable(name="enc_biases",shape=[units],initializer=tf.constant_initializer(0.0),dtype=tf.float32,regularizer=tf.contrib.layers.l2_regularizer(reg_lambda))
                                
            
    fc1_aux = tf.add(tf.matmul(corrupt_aux,ew),eb)
    Encoder_aux_item = tf.nn.leaky_relu(fc1_aux)               #leaky relu
    fc1_tar = tf.add(tf.matmul(corrupt_tar,ew),eb)
    Encoder_tar_item = tf.nn.leaky_relu(fc1_tar)

    dw = tf.transpose(ew)
    db = tf.get_variable(name="dec_biases",shape=[latent_dim],initializer=tf.constant_initializer(0.0),dtype=tf.float32,regularizer=tf.contrib.layers.l2_regularizer(reg_lambda))
                                    
            
    fc_aux = tf.add(tf.matmul(Encoder_aux_item,dw),db)
    Decoder_aux_item = tf.nn.leaky_relu(fc_aux)
    fc_tar = tf.add(tf.matmul(Encoder_tar_item,dw),db)
    Decoder_tar_item = tf.nn.leaky_relu(fc_tar)

    return Decoder_aux_item, Decoder_tar_item, Encoder_aux_item, Encoder_tar_item 


def FM_aux(hidden_neuron,Encoder_aux_user, Encoder_aux_item):
    fm_aux = tf.multiply(Encoder_aux_user, Encoder_aux_item)
    fm_aux = tf.nn.relu(fm_aux)
    wmul_aux = tf.Variable(tf.random_uniform([hidden_neuron,1],-0.1,0.1),name='wmul_aux')
    mul_aux = tf.matmul(fm_aux,wmul_aux)
    predict_rating_aux = tf.reduce_sum(mul_aux , 1,keep_dims=True)
    
    return predict_rating_aux

def FM_tar(hidden_neuron,Encoder_tar_user,Encoder_tar_item):
    fm = tf.multiply(Encoder_tar_user,Encoder_tar_item)
    fm = tf.nn.relu(fm)
    wmul = tf.Variable(tf.random_uniform([hidden_neuron,1],-0.1,0.1),name='wmul')
    mul = tf.matmul(fm,wmul)
    predict_rating_tar = tf.reduce_sum(mul , 1,keep_dims=True)
    
    return predict_rating_tar

def FM1_aux(latent_dim,Decoder_aux_user, Decoder_aux_item):
    fm_aux = tf.multiply(Decoder_aux_user, Decoder_aux_item)
    fm_aux = tf.nn.relu(fm_aux)
    wmul_aux = tf.Variable(tf.random_uniform([latent_dim,1],-0.1,0.1),name='wmul_aux')
    mul_aux = tf.matmul(fm_aux,wmul_aux)
    predict_rating1_aux = tf.reduce_sum(mul_aux , 1,keep_dims=True)
    
    return predict_rating1_aux

def FM1_tar(latent_dim,Decoder_tar_user,Decoder_tar_item):
    fm = tf.multiply(Decoder_tar_user,Decoder_tar_item)
    fm = tf.nn.relu(fm)
    wmul = tf.Variable(tf.random_uniform([latent_dim,1],-0.1,0.1),name='wmul')
    mul = tf.matmul(fm,wmul)
    predict_rating1_tar = tf.reduce_sum(mul , 1,keep_dims=True)
    
    return predict_rating1_tar



def update_p(loss1,loss2,pre_rec_loss_auxuser,pre_rec_loss_auxitem,loss_FM_aux1):
    
    pu = tf.ones_like(pre_rec_loss_auxuser,dtype=tf.float32)
    pv = tf.ones_like(pre_rec_loss_auxuser,dtype=tf.float32)
    zeros = tf.zeros_like(pre_rec_loss_auxuser,dtype=tf.float32)
    rev_loss= tf.ones_like(pre_rec_loss_auxuser,dtype=tf.float32)
    loss1 = loss1*rev_loss
    loss2 = loss2*rev_loss
    loss_u = pre_rec_loss_auxuser+loss_FM_aux1
    loss_v = pre_rec_loss_auxitem+loss_FM_aux1
    pu = tf.where(loss_u < loss1, pu, zeros)
    pv = tf.where(loss_u < loss2, pv, zeros)

    return pu,pv

def l2_norm(tensor):
    return tf.sqrt(tf.reduce_sum(tf.square(tensor)))


def train_model():
    users_aux = tf.placeholder(tf.int32, shape=[None])
    items_aux = tf.placeholder(tf.int32, shape=[None])
    users_inputs_aux = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    items_inputs_aux = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    ratings_aux = tf.placeholder(tf.float32, shape=[None,1])

    users_tar = tf.placeholder(tf.int32, shape=[None])
    items_tar = tf.placeholder(tf.int32, shape=[None])
    users_inputs_tar = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    items_inputs_tar = tf.placeholder(tf.int32, shape=[None, max_doc_length])
    ratings_tar = tf.placeholder(tf.float32, shape=[None,1])    

    
    dropout_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    rev_lossu = tf.placeholder(tf.float32)
    rev_lossv = tf.placeholder(tf.float32)

    text_embedding_aux = tf.Variable(word_embedding_mtrx_aux, dtype=tf.float32, name="review_text_embeds")
    padding_embedding = tf.Variable(np.zeros([1, word_latent_dim]), dtype=tf.float32)

    text_mask_aux = tf.constant([1.0] * text_embedding_aux.get_shape()[0] + [0.0])

    word_embeddings_aux = tf.concat([text_embedding_aux, padding_embedding], 0)
    word_embeddings_aux = word_embeddings_aux * tf.expand_dims(text_mask_aux, -1)

    text_embedding_tar = tf.Variable(word_embedding_mtrx_tar, dtype=tf.float32, name="review_text_embeds")

    text_mask_tar = tf.constant([1.0] * text_embedding_tar.get_shape()[0] + [0.0])

    word_embeddings_tar = tf.concat([text_embedding_tar, padding_embedding], 0)
    word_embeddings_tar = word_embeddings_tar * tf.expand_dims(text_mask_tar, -1)

                                      
    user_embeds_aux,user_embeds_tar = user_embeds_model(word_latent_dim, latent_dim, num_filters, window_size, users_inputs_aux,users_inputs_tar, word_embeddings_aux, word_embeddings_tar, dropout_rate)
    item_embeds_aux,item_embeds_tar = item_embeds_model(word_latent_dim, latent_dim, num_filters, window_size, items_inputs_aux, items_inputs_tar,word_embeddings_aux, word_embeddings_tar, dropout_rate)


    Decoder_aux_user, Decoder_tar_user, Encoder_aux_user, Encoder_tar_user = AutoRec_model_user(hidden_neuron,num_factor, num_users_tar,num_users_aux,user_embeds_aux,user_embeds_tar)
    Decoder_aux_item, Decoder_tar_item, Encoder_aux_item, Encoder_tar_item = AutoRec_model_item(hidden_neuron,num_factor, num_items_tar,num_items_aux,item_embeds_aux,item_embeds_tar)
 

    predict_rating1_aux = FM1_aux(latent_dim,Decoder_aux_user, Decoder_aux_item)
    predict_rating1_tar = FM1_tar(latent_dim,Decoder_tar_user, Decoder_tar_item)
    predict_rating_aux = FM_aux(hidden_neuron,Encoder_aux_user, Encoder_aux_item)
    predict_rating_tar = FM_tar(hidden_neuron,Encoder_tar_user, Encoder_tar_item)
    

    pre_rec_loss_auxuser = tf.reduce_sum(tf.abs(user_embeds_aux - Decoder_aux_user),1,keep_dims=True)#***********         
    pre_rec_loss_auxitem = tf.reduce_sum(tf.abs(item_embeds_aux - Decoder_aux_item),1,keep_dims=True)

    pre_rec_loss_taruser =(user_embeds_tar - Decoder_tar_user)  #***********    
    pre_rec_loss_taritem = (item_embeds_tar - Decoder_tar_item)
    
    loss_FM_aux1 = ((tf.squared_difference(ratings_aux,predict_rating_aux)))

    
    pu,pv = update_p(rev_lossu,rev_lossv,pre_rec_loss_auxuser,pre_rec_loss_auxitem,loss_FM_aux1)

    rec_loss_auxuser = tf.nn.l2_loss(tf.multiply(pu,(pre_rec_loss_auxuser)))   
    rec_loss_auxitem = tf.nn.l2_loss(tf.multiply(pv,(pre_rec_loss_auxitem)))
    rec_loss_taruser = tf.nn.l2_loss(pre_rec_loss_taruser)
    rec_loss_taritem = tf.nn.l2_loss(pre_rec_loss_taritem)
    
    
    loss_FM_aux = tf.nn.l2_loss(tf.multiply(pv,tf.multiply(pu,tf.subtract(predict_rating_aux,ratings_aux))))
    loss_FM_tar = tf.nn.l2_loss(tf.subtract(predict_rating_tar, ratings_tar))
    loss_FM1_aux = tf.nn.l2_loss(tf.multiply(pv,tf.multiply(pu,tf.subtract(predict_rating1_aux,ratings_aux))))
    loss_FM1_tar = tf.nn.l2_loss(tf.subtract(predict_rating1_tar, ratings_tar))

    loss_aux = loss_FM_aux + rec_loss_auxuser + rec_loss_auxitem + loss_FM1_aux
    loss_tar = loss_FM_tar + rec_loss_taruser + rec_loss_taritem + loss_FM1_tar
    loss = loss_aux + loss_tar- rev_lossu*tf.reduce_mean(pu) - rev_lossv*tf.reduce_mean(pv)
    #loss1 = loss_FM_aux + rec_loss_auxuser
    #loss2 = loss_FM_aux + rec_loss_auxitem

    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        best_mse, best_mae = 2.0, 2.0
        for e in range(epochs):
            t = time()
            Loss_aux1 = 0.0       
            Loss_aux2 = 0.0
            loss_total = 0.0
            count = 0.0
            for i in range(int(math.ceil(len(user_input_aux) / float(batch_size_aux)))):
                user_batch_aux, item_batch_aux, user_input_batch_aux, item_input_batch_aux, rates_batch_aux = get_train_instance_batch_change_aux(i, batch_size_aux,user_input_aux,item_input_aux, rateings_aux,user_reviews_aux,item_reviews_aux)
                user_batch_tar, item_batch_tar, user_input_batch_tar, item_input_batch_tar, rates_batch_tar = get_train_instance_batch_change_aux(i, batch_size_tar,user_input_tar,item_input_tar, rateings_tar,user_reviews_tar,item_reviews_tar)
               
               
                _, loss_val,Loss_aux,PU,PV,Pre_rec_loss_auxuser,Loss_FM_aux1,Predict_rating_aux,Ratings_aux,User_embeds_aux = sess.run([train_step, loss,loss_aux,pu,pv,pre_rec_loss_auxuser,loss_FM_aux1,predict_rating_aux,ratings_aux,Decoder_aux_user],
                                       feed_dict={users_aux: user_batch_aux, items_aux: item_batch_aux, users_inputs_aux: user_input_batch_aux, items_inputs_aux: item_input_batch_aux,ratings_aux: rates_batch_aux,users_tar: user_batch_tar, items_tar: item_batch_tar, users_inputs_tar: user_input_batch_tar, items_inputs_tar: item_input_batch_tar,ratings_tar: rates_batch_tar,
                                                  dropout_rate:drop_out,keep_prob :0.5,rev_lossu:2.0,rev_lossv:1.9})
                
                   
                Loss_aux1 += Loss_aux
                
                     
                loss_total += loss_val
                count += 1.0
            rev_loss1 = Loss_aux1/count
            rev_loss2 = Loss_aux1/count

            t1 = time()
            val_mses, val_maes = [], []

            for i in range(len(user_input_tar_val)):
                eval_model(users_tar, items_tar, users_inputs_tar, items_inputs_tar, dropout_rate,keep_prob, predict_rating_tar, sess, user_tar_vals[i],
                           item_tar_vals[i], user_input_tar_val[i], item_input_tar_val[i], rating_input_tar_val[i], val_mses, val_maes)
            val_mse = np.array(val_mses).mean()
            t2 = time()
            mses, maes = [], []
            for i in range(len(user_input_tar_test)):
                eval_model(users_tar, items_tar, users_inputs_tar, items_inputs_tar, dropout_rate,keep_prob, predict_rating_tar, sess, user_tar_tests[i], item_tar_tests[i], user_input_tar_test[i], item_input_tar_test[i], rating_input_tar_test[i], mses, maes)
            mse = np.array(mses).mean()
            mae = np.array(maes).mean()
            t3 = time()
            print( "epoch%d train time: %.3fs test time: %.3f  loss = %.5f val_mse = %.6f mse = %.6f mae = %.6f"%(e, (t1 - t), (t3 - t2), loss_total/count, val_mse, mse, mae))
            best_mse = mse if mse < best_mse else best_mse
            best_mae = mae if mae < best_mae else best_mae
        print("End. best_mse: %.6f, best_mae: %.6f" % (best_mse, best_mae))

def eval_model(users_tar, items_tar, users_inputs_tar, items_inputs_tar, dropout_rate,keep_prob, predict_rating_tar, sess, user_batch_tar, item_batch_tar, user_input_batch_tar, item_input_batch_tar, rate_tests, rmses, maes):

    predicts = sess.run(predict_rating_tar, feed_dict={users_tar: user_batch_tar, items_tar: item_batch_tar, users_inputs_tar: user_input_batch_tar, items_inputs_tar: item_input_batch_tar, dropout_rate:1.0,keep_prob :0.5})
    row, col = predicts.shape
    for r in range(row):
        rmses.append(pow((predicts[r, 0] - rate_tests[r][0]), 2))
        maes.append(abs((predicts[r, 0] - rate_tests[r][0])))
    return rmses, maes

if __name__ == "__main__":

    word_latent_dim = 300
    latent_dim = 32
    max_doc_length = 300
    num_filters = 16
    window_size = 3
    num_factor = 32
    hidden_neuron = 100
    
    learning_rate = 0.002
  
    drop_out = 0.8
    n_hidden = 1024
    
    epochs = 200
    batch_size_aux = 298
    batch_size_tar = 518
    
    print("1.6*******,dim=30")

    firTime = time()
    dataSet_aux = Dataset_aux(max_doc_length, "./data/instant_video/","WordDict.out")
    dataSet_tar = Dataset_tar(max_doc_length, "./data/music/","WordDict.out")

    word_dict_aux, user_reviews_aux, item_reviews_aux, train_aux,word_dict_tar, user_reviews_tar, item_reviews_tar, train_tar, valRatings, testRatings = dataSet_aux.aux_word_id_dict, dataSet_aux.aux_userReview_dict, dataSet_aux.aux_itemReview_dict, dataSet_aux.aux_trainMtrx, dataSet_tar.tar_word_id_dict, dataSet_tar.tar_userReview_dict, dataSet_tar.tar_itemReview_dict, dataSet_tar.tar_trainMtrx,dataSet_tar.valRatings, dataSet_tar.testRatings
    secTime = time()

    num_users_aux, num_items_aux = train_aux.shape
    num_users_tar, num_items_tar = train_tar.shape
    print ("load data: %.3fs" % (secTime - firTime))
    print ("num_users_tar, num_items_tar",num_users_tar, num_items_tar)
    print ("num_users_aux, num_items_aux",num_users_aux, num_items_aux)
    print ( latent_dim)

    #load word embeddings
    word_embedding_mtrx_aux = ini_word_embed_aux(len(word_dict_aux), word_latent_dim)
    word_embedding_mtrx_tar = ini_word_embed_tar(len(word_dict_tar), word_latent_dim)

    print ("shape", word_embedding_mtrx_aux.shape)
    print ("shape", word_embedding_mtrx_tar.shape)
    # get train instances
    user_input_aux, item_input_aux, rateings_aux = get_train_instance_aux(train_aux)
    user_input_tar, item_input_tar, rateings_tar = get_train_instance_aux(train_tar)
    print ("len(user_input_aux), len(item_input_aux), len(rateings_aux)",len(user_input_aux), len(item_input_aux), len(rateings_aux))
    print ("len(user_input_tar), len(item_input_tar), len(rateings_tar)",len(user_input_tar), len(item_input_tar), len(rateings_tar))
    # get test/val instances
    user_tar_vals, item_tar_vals, user_input_tar_val, item_input_tar_val, rating_input_tar_val = get_test_list(200, valRatings, user_reviews_tar, item_reviews_tar)
    user_tar_tests, item_tar_tests, user_input_tar_test, item_input_tar_test, rating_input_tar_test = get_test_list(200, testRatings, user_reviews_tar, item_reviews_tar)

    #train & eval model
    train_model()
