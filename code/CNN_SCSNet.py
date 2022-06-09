# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:24:34 2018

@author: ZHENGUO
"""
#768-2-768
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import pandas as pd
import numpy as np
np.random.seed(0)
#import matplotlib.pyplot as plt
import tensorflow as tf
import time

def main(argv = None):
    #file_name_reference = '../data/rectangle_s.npy'
    #save_prefix_whereToSave = '../result/trained_models_rectangle_single/'
    file_name_reference = '../data/all_data_s.npy'
    save_prefix_whereToSave = '../result/trained_models_single/'

    # The batch size defines the number of samples that will be propagated through the network. # For instance, let's say you have 1050 training samples and you want to set up a batch_size equal to 100. The algorithm takes the first 100 samples (from 1st to 100th) from the training dataset and trains the network. Next, it takes the second 100 samples (from 101st to 200th) and trains the network again.
    batch_size = 256
    # Step Decay is a learning rate schedule that drops the learning rate by a factor every few epochs
    # The number of steps between each decay
    decay_steps = 500
    # The amount decayed
    decay_rate = 0.98
    # Initial learing rate (initial name: starting_learning_rate)
    initial_learning_rate = 1e-3
    train_ratio = 0.99
    n_epochs = 5001
    # ??? Why have this and 'initial_learning_rate'  (seems to change the number of rows shown, sample_rate=1 shows all rows, any less reduces the rows shown)
    sample_rate = 1.0

    #define the size of graph
    height = 24
    width = 32
    resolution = height*width
    all_file_data = np.array(np.load(file_name_reference)).astype(np.float32)
    # All of these only use the columns from 770 - 1538
    stress_mean = np.mean(all_file_data[:,770:1538])
    stress_min = np.min(all_file_data[:,770:1538])
    stress_max = np.max(all_file_data[:,770:1538])
    stree_statistic = [['mean','max','min'],[stress_mean,stress_min,stress_max]]

    # This looks like the stuff to make the graph
    '''
    pickup_row = 20
    plt_row_node = all_file_data[pickup_row,0:resolution].reshape(24,32)
    plt_row_load = all_file_data[pickup_row,resolution:resolution+2]
    plt_row_stress = all_file_data[pickup_row,resolution+2:resolution*2+2].reshape(24,32)
    im_node = plt.imshow(plt_row_node,cmap='jet',interpolation='nearest')
    plt.title(r'$q(x)=$'+str(round(plt_row_load[0],1))+r'$N/mm^2,$'+r'$ q(y)=$'+str(round(plt_row_load[1],1))+r'$N/mm^2$', fontsize=15)
    plt.show()
    
    im_stress = plt.imshow(plt_row_stress,cmap='jet',interpolation='nearest')
    plt.title(r'$q(x)=$'+str(round(plt_row_load[0],1))+r'$N/mm^2,$'+r'$ q(y)=$'+str(round(plt_row_load[1],1))+r'$N/mm^2$', fontsize=15)
    plt.colorbar(im_stress)
    plt.show()
    '''
    # seems like it just changes the position of subarrays. Each row in the dataframe looking thing is an array. ??? why do this
    np.random.shuffle(all_file_data)
    all_file_data=all_file_data[:int(sample_rate * np.shape(all_file_data)[0]),:]

    #train/test split
    # This is a single integer equal to the number of rows multiplied to the train_ratio.   This separate the training samples form the testing samples with the ratio given by the train_ratio
    num_train = int(train_ratio * np.shape(all_file_data)[0])
    #num_test = np.shape(all_file_data)[0]-num_train

    # uses train/test split numbers to create 2 numpy arrays with the train and test data respecitively
    np_data_train = all_file_data[:num_train, :]
    np_data_test = all_file_data[num_train:, :]
    
    #CNN
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return(tf.Variable(initial))
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return(tf.Variable(initial))
        
    def conv2d(x,W):
        return(tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME'))
        
    def max_pool_2x2(x):
        return(tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))
    
    
    #define placeholders for xs,ys,keep_prob
    xs_node = tf.placeholder(tf.float32, shape=[None,resolution],name='xs_node')
    xs_reshape = tf.reshape(xs_node, shape=[-1,height,width,1])
    xs_load = tf.placeholder(tf.float32, shape=[None,2],name='xs_load')
    ys = tf.placeholder(tf.float32, shape=[None,resolution],name='ys')
    keep_prob = tf.placeholder(tf.float32)
    
    #Define the 1st CNN layer for Eecoding and Pooling
    W_conv1 = weight_variable([3,3,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(xs_reshape,W_conv1)+b_conv1) #Nonex24x32x32
    h_pool1 = max_pool_2x2(h_conv1) #Nonex12x16x32
    
    #Define the 2nd CNN layer for Eecoding and Pooling
    W_conv2 = weight_variable([3,3,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #Nonex12x16x64
    h_pool2 = max_pool_2x2(h_conv2) #6x8x64
    
    #define FC1
    W_fc1 = weight_variable([6*8*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1,6*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    
    #define FC2
    W_fc2 = weight_variable([1024, 30]) 
    b_fc2 = bias_variable([30])
    h_fc2 = tf.nn.softplus(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  #Nonex30 的向量
    #h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  #Nonex30 的向量
    
    #Feature representation fully connected
    dna_full = tf.concat([h_fc2,xs_load],1)
    #None x32 的向量
    
    #Define FC3
    W_fc3 = weight_variable([32, 1024]) 
    b_fc3 = bias_variable([1024])
    h_fc3 = tf.nn.softplus(tf.matmul(dna_full,W_fc3)+b_fc3) #None x1024 的向量
    #h_fc3 = tf.nn.relu(tf.matmul(dna_full,W_fc3)+b_fc3) #None x1024 的向量
    
    #Define FC4
    W_fc4 = weight_variable([1024, 6*8*64])
    b_fc4 = bias_variable([6*8*64])
    h_fc4 = tf.nn.relu(tf.matmul(h_fc3,W_fc4)+b_fc4) #Nonex3072 的向量
    h_fc4_drop = tf.nn.dropout(h_fc4,keep_prob)
    
    #Define Upsampling and the 3rd CNN layer for Decoding
    #reshape
    h_fc4_drop_flat = tf.reshape(h_fc4_drop, [-1,6,8,64]) #Nonex6x8x64 的向量
    h_fc4_up = tf.keras.layers.UpSampling2D(size=(2,2))(h_fc4_drop_flat) #Nonex12x16x64
    W_conv3 = weight_variable([3,3,64,32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_fc4_up,W_conv3)+b_conv3) #Nonex12x16x32
    
    #Define Upsampling and the 4th CNN layer for Decoding
    h_conv3_up = tf.keras.layers.UpSampling2D(size=(2,2))(h_conv3) #Nonex24x32x32
    W_conv4 = weight_variable([3,3,32,16])
    b_conv4 = bias_variable([16])
    h_conv4 = tf.nn.relu(conv2d(h_conv3_up,W_conv4)+b_conv4) #Nonex24x32x16
    
    #Define the 5th CNN layer for Decoding
    W_conv5 = weight_variable([3,3,16,1])
    b_conv5 = bias_variable([1])
    prediction_matrix = tf.nn.relu(conv2d(h_conv4,W_conv5)+b_conv5) #Nonex24x32x1
    prediction = tf.reshape(prediction_matrix, shape=[-1,resolution])#Nonex768
    
    
    #metrics
    mse = tf.losses.mean_squared_error(ys,prediction)
    mae = tf.reduce_mean(tf.abs(tf.subtract(ys,prediction)))
    #loss
    loss = mse
    
    
    #train rate   
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_steps,decay_rate=decay_rate)
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    
    #set GPU
    config = tf.ConfigProto() 
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    
    
    '''
    #set CPU
    config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage  
                    inter_op_parallelism_threads = 1,   
                    intra_op_parallelism_threads = 1,  
                    log_device_placement=True)
    '''
    
    saver = tf.train.Saver()    
    start_time = time.localtime()
    print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
    
    #session
    with tf.Session(config = config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        test_num = 300
        mse_train = np.zeros(n_epochs)
        mse_test = np.zeros(n_epochs)
        mae_train = np.zeros(n_epochs)
        mae_test = np.zeros(n_epochs)
        prediction_num = 50
        prediction_history = np.zeros((prediction_num,resolution,n_epochs+1))
        prediction_history[:,:,0] = np_data_test[0:prediction_num,770:1538]        
        print('Training...')
        for epoch in range(n_epochs):
            print(str(epoch)+' epoch of '+str(n_epochs))
            
            for iteration in range(num_train // batch_size):
                unit_iter = (num_train // batch_size)//6
                if iteration % unit_iter == 0:
                    print('   '+str(iteration)+' iteration of '+str(num_train // batch_size))
                #These are the training and testing stuff        32x24 = 768.
                x_batch_node = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,0:768]
                x_batch_load = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,768:770]
                #This is our target stuff     1538-770=768   also a 32x24
                y_batch = np_data_train[iteration*batch_size:iteration*batch_size+batch_size,770:1538]

                _, l_rate = sess.run([add_global,learning_rate], feed_dict={xs_node:x_batch_node, xs_load:x_batch_load, ys:y_batch, keep_prob:0.5})
                sess.run(train_step, feed_dict={xs_node:x_batch_node, xs_load:x_batch_load, ys:y_batch, keep_prob:0.5})
                
            print('Epoch:',epoch,', Learning rate:',l_rate)
                                    
            [mse_train[epoch], mae_train[epoch]] = sess.run([mse,mae], feed_dict={xs_node:x_batch_node, xs_load:x_batch_load, ys:y_batch, keep_prob:0.5})
            print('Epoch:',epoch, 'mse_train:', mse_train[epoch])  
            print('Epoch:',epoch, 'MAE_train:', mae_train[epoch])
            
            [mse_test[epoch], mae_test[epoch],prediction_temp] =  sess.run([mse,mae,prediction], feed_dict={xs_node:np_data_test[0:test_num,0:768], xs_load:np_data_test[0:test_num,768:770], ys:np_data_test[0:test_num,770:1538], keep_prob:0.5})
            prediction_history[:,:,epoch+1] = prediction_temp[0:prediction_num,:]
            print('Epoch:',epoch, 'mse_test:', mse_test[epoch])
            print('Epoch:',epoch, 'MAE_test:', mae_test[epoch])
                        
            if epoch%1000 == 0:
                save_path = save_prefix_whereToSave
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_file = save_prefix_whereToSave + 'cnn_single'
                saver.save(sess,save_file,global_step=epoch,write_meta_graph=True)
                print('Session is saved: {}'.format(save_path))
    
            current_time = time.localtime()
            print('Time current: ', time.strftime('%Y-%m-%d %H:%M:%S', current_time))

        print('Training is finished!')
    
    np.save(save_prefix_whereToSave+'mse_train',mse_train)
    np.save(save_prefix_whereToSave+'mse_test',mse_test)
    np.save(save_prefix_whereToSave+'mae_train',mae_train)
    np.save(save_prefix_whereToSave+'mae_test',mae_test)
    np.save(save_prefix_whereToSave+'prediction_history',prediction_history)
    np.save(save_prefix_whereToSave+'stree_statistic',stree_statistic)
    
    print('Mean stress is:',stress_mean)
    end_time = time.localtime()
    print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
    print('Time end: ', time.strftime('%Y-%m-%d %H:%M:%S', end_time))
    
if __name__ == '__main__':
    main()