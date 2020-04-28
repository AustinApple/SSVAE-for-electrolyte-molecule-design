# this is the run script for our own data(MP and zinc), try to predict IE and EA 
from __future__ import print_function
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
import numpy as np 
import sys
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
from preprocessing import smiles_to_seq, vectorize
import SSVAE_reload

from preprocessing import get_property, canonocalize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


beta=10000.
 
# include only fourty 39 characters
char_set=[" ", "@", "H", "N", "S", "o", "i", "6", "I", "]", "P", "5", ")", "4", "8", "B", "F", 
           "3", "9", "c", "-", "2", "p", "0", "n", "C", "(", "=", "+", "#", "1", "/", "7", 
           "s", "O", "[", "Cl", "Br", "\\"] 




#data_uri='./data/ZINC_310k.csv'
data_MP = './MP_new_canonize_cut.csv'
data_zinc = './data_size_test/30W.csv'

save_uri='./test_MP_zinc_40W.ckpt'

# I think the smiles in last 2000 smiles has problem
frac_val=0.1      # fraction for valuation  0.01 ok 0.1 ok
ntst=2000           #  number of data for test in MP  ==1000 and zinc 300000 == problem
n_zinc=300000 #  number of zinc data (unlabel data)

data = pd.read_csv(data_MP)

np.random.seed(0)
random_id = np.random.permutation(data.shape[0])
# np.save("seed_1_test_idx",random_id[-ntst:])

data = data.iloc[random_id,:]

smiles_MP = data.as_matrix()[:,0] #0: SMILES
smiles_zinc = pd.read_csv(data_zinc).as_matrix()[:n_zinc,0]
smiles = np.concatenate((smiles_MP, smiles_zinc), axis=0)
    

# data preparation
print('::: data preparation')


Y = np.asarray(data.as_matrix()[:,1:], dtype=np.float32)  # 1.IE   2.EA 

end_index_MP = smiles_MP.shape[0]   # before this number is MP, after is zinc 
# data preparation

list_seq = smiles_to_seq(smiles, char_set)


Xs, X=vectorize(list_seq, char_set)

tstX=X[end_index_MP-ntst:end_index_MP]
tstXs=Xs[end_index_MP-ntst:end_index_MP]
tstY=Y[end_index_MP-ntst:end_index_MP]


nL=int(len(Y)-ntst)       # subtract the number of test set 
nU=int(n_zinc)            # symbol the number of zinc data (unlabeled)  
nL_trn=int(nL*(1-frac_val))  
nL_val=int(nL*frac_val)
nU_trn=int(nU*(1-frac_val))
nU_val=int(nU*frac_val)

perm_id_nL=np.random.permutation(nL)
perm_id_nU=np.random.permutation([i for i in range(end_index_MP,len(X))])
# do not do permutation
# perm_id_nL=np.array([i for i in range(nL)])
# perm_id_nU=np.array([i for i in range(end_index_MP,len(X))])
print(perm_id_nU[:10])
print(perm_id_nU[:10])

trnX_L=X[perm_id_nL[:nL_trn]]
trnXs_L=Xs[perm_id_nL[:nL_trn]]
trnY_L=Y[perm_id_nL[:nL_trn]]

valX_L=X[perm_id_nL[nL_trn:nL_trn+nL_val]]
valXs_L=Xs[perm_id_nL[nL_trn:nL_trn+nL_val]]
valY_L=Y[perm_id_nL[nL_trn:nL_trn+nL_val]]

trnX_U=X[perm_id_nU[:nU_trn]]
trnXs_U=Xs[perm_id_nU[:nU_trn]]

valX_U=X[perm_id_nU[nU_trn:nU_trn+nU_val]]
valXs_U=Xs[perm_id_nU[nU_trn:nU_trn+nU_val]]

scaler_Y = StandardScaler()
scaler_Y.fit(Y)
trnY_L=scaler_Y.transform(trnY_L)
valY_L=scaler_Y.transform(valY_L)

## model training
print('::: model training')

seqlen_x = X.shape[1]
dim_x = X.shape[2]
dim_y = Y.shape[1]
dim_z = 100
dim_h = 250

n_hidden = 3
batch_size = 200

model = SSVAE_reload.Model(seqlen_x = seqlen_x, dim_x = dim_x, dim_y = dim_y, dim_z = dim_z, dim_h = dim_h,
                    n_hidden = n_hidden, batch_size = batch_size, beta = float(beta), char_set = char_set)

with model.session:
    model.train(trnX_L=trnX_L, trnXs_L=trnXs_L, trnY_L=trnY_L, trnX_U=trnX_U, trnXs_U=trnXs_U,
                valX_L=valX_L, valXs_L=valXs_L, valY_L=valY_L, valX_U=valX_U, valXs_U=valXs_U)
    
    # model.saver.save(model.session, save_uri)
    # model.predict_train(x_input=trnX_L,y_input=trnY_L)
    
    # property prediction performance
   
    tstY_hat=scaler_Y.inverse_transform(model.predict(tstX))
    

    for j in range(dim_y):
        print([j, mean_absolute_error(tstY[:,j], tstY_hat[:,j])])


    # model.reload(trnX_L=trnX_L, trnXs_L=trnXs_L, trnY_L=trnY_L, trnX_U=trnX_U, trnXs_U=trnXs_U,
    #             valX_L=valX_L, valXs_L=valXs_L, valY_L=valY_L, valX_U=valX_U, valXs_U=valXs_U)
    
    # model.predict_train(x_input=trnX_L,y_input=trnY_L)
    # tstY_hat=scaler_Y.inverse_transform(model.predict(tstX))
    
    # for j in range(dim_y):
    #     print([j, mean_absolute_error(tstY[:,j], tstY_hat[:,j])])
        
        
    # ## unconditional generation
    # for t in range(10):
    #     smi = model.sampling_unconditional()
    #     print([t, smi])
    #     # print([t, smi, get_property(smi)])
    
    # ## conditional generation (e.g. MolWt=250)
    # yid = 0
    # ytarget = 5.0
    # ytarget_transform = (ytarget-scaler_Y.mean_[yid])/np.sqrt(scaler_Y.var_[yid])
    
    # print('this is for conditional sampling')

    # for t in range(100):

    #     smi = model.sampling_conditional(yid, ytarget_transform)
        
    #     # print([t, smi, get_property(smi)])
    #     print([t, smi])