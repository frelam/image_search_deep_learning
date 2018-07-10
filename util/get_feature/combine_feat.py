import numpy as np
a = np.load('/home/scw4750/frelam_20161027/get_feature/data/feature_0w-5w_norm_2rd.npy')
b = np.load('/home/scw4750/frelam_20161027/get_feature/data/feature_5w-10w_norm_2rd.npy')
c = np.load('/home/scw4750/frelam_20161027/get_feature/data/feature_10w-15w_norm_2rd.npy')
a = np.vstack((a,b))
a = np.vstack((a,c))
f = '/home/scw4750/frelam_20161027/get_txt/feat_2rd.npy'
np.save(f,a)
