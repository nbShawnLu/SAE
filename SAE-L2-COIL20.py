import tensorflow as tf
import scipy.io as sio
import os

from AEnet import ConvAE
from AEutils import  *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


d = 12
alpha = 8
ro = 0.04

data = sio.loadmat('./Data//COIL20.mat')
Img = data['fea']
Label = data['gnd']
Img = np.reshape(Img,(Img.shape[0],32,32,1))

n_input = [32,32]
kernel_size = [3]
n_hidden = [15]
batch_size = 20*72
model_path = './pretrain-model-COIL20/model.ckpt'
ft_path = './pretrain-model-COIL20/model.ckpt'
logs_path = './logs'

num_class = 20 #how many class we sample
num_sa = 72

batch_size_test = num_sa * num_class


iter_ft = 0
ft_times = 110
display_step = 400

fine_step = -1

reg1 = 1.0e-2
reg02 = 100
reg03 = 1e-5

mm = 0
mreg2 = 0
mreg3 = 0
mlr2 = 0
startfrom = [0,0,0]
mm= 0

for reg2 in [2e-2,1e-1,5e-1,1,2,3,4,5,6,10,20,30,40,50,60,100]:
	for reg3 in [0.1,0.2,0.5,1,2,3,4,5,6,10,20,30,50,80]:
		for learning_rate in [1e-4, 2e-4, 3e-4, 4e-4,1e-3]:
			try:
				if reg2<startfrom[0] or (reg2==startfrom[0] and reg3<startfrom[1]) or (reg2==startfrom[0] and reg3==startfrom[1] and learning_rate<startfrom[2]):
					continue
				print("reg:", reg2, reg3,learning_rate)
				tf.reset_default_graph()
				CAE = ConvAE(n_input = n_input, n_hidden = n_hidden, reg_constant1 = reg1, re_constant2 = reg2, re_constant3 = reg3, kernel_size = kernel_size, \
							batch_size = batch_size_test, model_path=model_path, restore_path=model_path, logs_path= logs_path)

				acc_= []
				for i in range(0,1):
					coil20_all_subjs = Img
					coil20_all_subjs = coil20_all_subjs.astype(float)
					label_all_subjs = Label
					label_all_subjs = label_all_subjs - label_all_subjs.min() + 1
					label_all_subjs = np.squeeze(label_all_subjs)

					CAE.initlization()
					CAE.restore()
					COLD = None
					lastr = 1.0
					losslist = []
					for iter_ft  in range(ft_times):
						cost, C = CAE.partial_fit(coil20_all_subjs, learning_rate, mode='fine')  #
						losslist.append(cost[-1])
						if iter_ft % display_step == 0 and iter_ft > 10:
							print ("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0]/float(batch_size_test)))
							print(cost)
							for posti in range(2):
								display(C, coil20_all_subjs, d, alpha, ro, num_class, label_all_subjs)

						if COLD is not None:
							normc = np.linalg.norm(COLD,ord='fro')
							normcd =np.linalg.norm(C-COLD,ord='fro')
							r = normcd / normc
							# print(epoch,r)
							if r < 3.0e-4 and lastr < 3.0e-4:
								print("early stop")
								print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
								print(cost)
								for posti in range(2):
									display(C, coil20_all_subjs, d, alpha, ro, num_class, label_all_subjs)
								break
							lastr = r
						COLD = C

					print("epoch: %.1d" % iter_ft, "cost: %.8f" % (cost[0] / float(batch_size_test)))
					print(cost)

					for posti in range(3):
						drawC(C)
						acc = display(C, coil20_all_subjs, d, alpha, ro, num_class, label_all_subjs)
						acc_.append(acc)
					acc_.append(acc)

				acc_ = np.array(acc_)
				print(acc_)
				lossnp = np.asarray(losslist)
				#np.savetxt("loss-l2.csv", lossnp, delimiter=',')
				if max(acc_) > mm:
					mreg2 = reg2
					mreg3 = reg3
					mlr2 = learning_rate
					mm = max(acc_)
				print("max:", mreg2, mreg3, mlr2, mm)
			except:
				print("error in ", reg2, reg3,learning_rate)
			finally:
				try:
					CAE.sess.close()
				except:
					''



