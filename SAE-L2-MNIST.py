import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from tensorflow.examples.tutorials.mnist import input_data

from AEnet import ConvAE
from AEutils import  *

# SELECT GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_face(Img, Label, CAE, num_class,lr2=5e-4):

    d = 3
    alpha = 1
    ro = 0.06
    
    acc_= []
    for i in range(0,11-num_class):
        face_10_subjs = np.array(Img[100*i:100*(i+num_class),:])
        face_10_subjs = face_10_subjs.astype(float)        
        label_10_subjs = np.array(Label[100*i:100*(i+num_class)])
        label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
        label_10_subjs = np.squeeze(label_10_subjs) 
                     
        CAE.initlization()        
        CAE.restore() # restore from pre-trained model    
        
        max_step = 80#50 + num_class*25# 100+num_class*20
        display_step = 200#max_step/20#10
        # fine-tune network
        epoch = 0
        COLD = None
        lastr = 1.0
        while epoch < max_step:
            epoch = epoch + 1
            cost, Coef = CAE.partial_fit(face_10_subjs, lr2, mode = 'fine')  #
            if epoch % display_step == 0:
                print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0]/float(batch_size))   )
                print(cost)
                for posti in range(2):
                    display(Coef, label_10_subjs, d, alpha, ro)


            if COLD is not None:
                normc = np.linalg.norm(COLD, ord='fro')
                normcd = np.linalg.norm(Coef - COLD, ord='fro')
                r = normcd/normc
                #print(epoch,r)
                if r < 5.0e-4 and lastr < 5.0e-4 and epoch < fine_step:
                    fine_step = epoch
                    print("fine step = ",fine_step)
                if r < 1.0e-8 and lastr < 1.0e-8 and epoch >  fine_step:
                    print("early stop")
                    print("epoch: %.1d" % epoch, "cost: %.8f" % (cost[0] / float(batch_size)))
                    print(cost)
                    for posti in range(5):
                        display(Coef, label_10_subjs, d, alpha, ro)
                    break
                lastr = r
            COLD = Coef

        for posti in range(5):
            drawC(Coef)
            acc_x = display(Coef, label_10_subjs, d, alpha, ro)
            acc_.append(acc_x)
        acc_.append(acc_x)    
    
    acc_ = np.array(acc_)
    mm = np.max(acc_)

    print("%d subjects:" % num_class)    
    print("Max: %.4f%%" % ((1-mm)*100))
    print(acc_) 
    
    return (1-mm)
    
   
        
    
if __name__ == '__main__':
    
    # load face images and labels
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    Img = []
    Label = []
    num = mnist.train.num_examples
    rawImg = mnist.train._images
    rawLabel = mnist.train._labels
    for i in range(10):
        ind = [ii for ii in range(num) if rawLabel[ii] == i]
        ind = ind[0:100]
        if i == 0:
            Img = rawImg[ind]
            Label = rawLabel[ind]
        else:
            Img = np.concatenate([Img,rawImg[ind]])
            Label =  np.concatenate([Label,rawLabel[ind]])
    Label = np.reshape(Label,(-1,1))

    model_path = './models/model-mnist.ckpt'
    restore_path = './models/model-mnist.ckpt'
    logs_path = './logs'

    # face image clustering
    n_input = [28, 28]
    n_hidden = [20, 10, 5]
    kernel_size = [5,3,3]
    
    Img = np.reshape(Img,[Img.shape[0],n_input[0],n_input[1],1]) 
    
    all_subjects = [10]
    reg1 = 1.0
    reg02 = 2
    reg03 = 1e-1

    mm = 0
    mreg2 = 0
    mreg3 = 0
    mlr2 = 0

    startfrom = [0, 0, 0]
    mm = 0

    for reg2 in [1e-2,5e-2,1e-1,1, 2,5,6,10,20,50,100,200]:
        for reg3 in [1e-2,2e-2,5e-2,1e-1,0.2,1,10,20,30,50,100,200]:
            for lr2 in [2e-4]:
                try:
                    print("reg:", reg2, reg3, lr2)
                    avg = []
                    med = []
                    iter_loop = 0
                    while iter_loop < len(all_subjects):
                        num_class = all_subjects[iter_loop]
                        batch_size = num_class * 100

                        tf.reset_default_graph()
                        CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, re_constant3=reg3, \
                                     kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=restore_path, logs_path=logs_path)

                        avg_i = test_face(Img, Label, CAE, num_class,lr2)
                        avg.append(avg_i)
                        iter_loop = iter_loop + 1
                        #visualize(Img, Label, CAE)
                    iter_loop = 0

                    if 1-avg[0] > mm:
                        mreg2 = reg2
                        mreg3 = reg3
                        mlr2 = lr2
                        mm = 1-avg[0]
                    print("max:", mreg2, mreg3, mlr2, mm)
                except:
                    print("error in ", reg2, reg3, lr2)
                finally:
                    try:
                        CAE.sess.close()
                    except:
                        ''

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
