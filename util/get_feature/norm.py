import cudamat as cm
import numpy as np
cm.cuda_set_device (0)
cm.cublas_init()
t = np.load('/home/scw4750/frelam_20161027/get_feature/data/feature_0w-5w.npy')
t.dtype = '<f'
feat = t[0:40000]

print t
a = cm.CUDAMatrix(feat)
c = cm.dot(a, a.T)
e = cm.sqrt(c)
e = e.asarray()
#e.dtype = 'float'
print len(e)
dioa = None
for index,item in enumerate(e):
    if dioa is None:
        temp = np.array(item[index])
        dioa = np.copy(temp)
    else:
        temp = np.array(item[index])
        dioa = np.vstack((dioa,temp))
feat = t[40000:50000]

a = cm.CUDAMatrix(feat)
c = cm.dot(a, a.T)
e_2 = cm.sqrt(c)
e_2 = e_2.asarray()
print len(e_2)
for index,item in enumerate(e_2):
    temp = np.array(item[index])
    dioa = np.vstack((dioa,temp))
feat = t
for index,item in enumerate(feat):
    
    if dioa[index][0]==0:
         continue
    feat[index,:]=item/(dioa[index][0])
print feat 
print dioa
f = '/home/scw4750/frelam_20161027/get_feature/data/feature_0w-5w_norm_2rd.npy'
np.save(f,feat)

