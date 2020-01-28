#!/usr/bin/env python2
# Copyright (c) 2016-2020, LVHAOYU.  All rights reserved.

"""
Classify an image using individual model files

Use this script as an example to build your own tool
"""
import cudamat as cm
import argparse
import os
import time
import Image
from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc
import socket
os.environ['GLOG_minloglevel'] = '2'  # Suppress most caffe output
import caffe  # noqa
from caffe.proto import caffe_pb2  # noqa


def get_net(caffemodel, deploy_file, use_gpu=True):
    """
    Returns an instance of caffe.Net

    Arguments:
    caffemodel -- path to a .caffemodel file
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    use_gpu -- if True, use the GPU for inference
    """
    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)


def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
    deploy_file -- path to a .prototxt file

    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(inputs={'data': dims})
    t.set_transpose('data', (2, 0, 1))  # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2, 1, 0))

    if mean_file:
        # set mean pixel
        with open(mean_file, 'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t


def load_image(path, height, width, mode='RGB'):
    """
    Load an image from disk

    Returns an np.ndarray (channels x width x height)

    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension

    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = PIL.Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image


def forward_pass(images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:, :, np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        feat = net.blobs['fc7'].data[0]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    return scores,feat


def read_labels(labels_file):
    """
    Returns a list of strings

    Arguments:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

def loadnet(caffemodel, deploy_file,mean_file=None, labels_file=None,use_gpu=True):
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
   
    return net,transformer
def classify(net,transformer,image_files,batch_size=None):
    """
    Classify some images against a Caffe model and print the results

    Arguments:
    caffemodel -- path to a .caffemodel
    deploy_file -- path to a .prototxt
    image_files -- list of paths to images

    Keyword arguments:
    mean_file -- path to a .binaryproto
    labels_file path to a .txt file
    use_gpu -- if True, run inference on the GPU
    """
    # Load the model and images
    # Classify the image
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = [load_image(image_file, height, width, mode) for image_file in image_files]
    scores,feat = forward_pass(images, net, transformer,batch_size)

    #
    # Process the results
    #

    indices = (-scores).argsort()[:, :5]  # take top 5 results
    #classifications = []
    return indices,feat
    '''
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # 'i' is a category in labels and also an index into scores
            if labels is None:
                label = 'Class #%s' % i
            else:
                label = labels[i]
            result.append((label, round(100.0*scores[image_index, i], 4)))
        classifications.append(result)

    for index, classification in enumerate(classifications):
        print '{:-^80}'.format(' Prediction for %s ' % image_files[index])
        for label, confidence in classification:
            print '{:9.4%} - "{}"'.format(confidence/100.0, label)
        print
    '''
def normolize(feat):
    """
    normolize the feature with gpu with cuda
    """
    #feat_temp = np.vstack((feat, feat))
    feat = np.reshape(feat,(4096,1))
    a = cm.CUDAMatrix(feat)
    c = cm.dot(a.T, a)
    c =cm.sqrt(c)
    c = c.asarray()
    feat = feat/c[0]
    '''
    for index,item in enumerate(feat):
        feat[index,:]=item/(c[index][index])
    '''
    return feat 

def search(indices,feat,feature_map,ID_map):
    feature_table=None
    ID_table=None
    indices = indices[0][0:2]
    a=27
    indices = np.hstack((indices,a))
    #print indices
    #print feature_map[0][0],ID_map[0][0:4]
    a = cm.CUDAMatrix(feat)
    for category in indices:
        if feature_table is None:
            #print category
            if (feature_map[category]):
                d = cm.dot(feature_map[category], a)
                f = d.asarray()
                feature_table = np.copy(f)
            if (ID_map[category]):
                ID_table = np.copy(ID_map[category]) 
        else:
            if (feature_map[category]):
                d = cm.dot(feature_map[category], a)
                f = d.asarray()
                feature_table = np.vstack((feature_table, f))
            if (ID_map[category]):
                ID_table = np.hstack((ID_table, ID_map[category]))
    #print feature_table[1]
    #ID_table = np.hstack((ID_table, ID_zl_array))
    '''
    #print feat
    c = cm.CUDAMatrix(feature_table)
    d = cm.dot(c, a)
    q = cm.dot(feature_zl_cm, a)
    t = d.asarray()
    r = q.asarray()
    e = np.vstack((t,r))
    '''
    #print e
    ind = np.argsort(-feature_table,axis=0)
    ind = ind [0:200]
    #print ind
    ID_result=ID_table[ind]
    
    '''
    for index in ind:
        if ID_result is None:
            ID_result = np.copy(ID_map[index]) 
        else:
            ID_result = np.hstack((ID_result, ID_map[index]))
    '''
    return  ID_result

def loaddatabase(feat_npy,num_npy,categories):
    feat_array = np.load(feat_npy)
    feat_array = feat_array[0:350000]
    feat_list = feat_array.tolist()
    num_array = np.load(num_npy)
    num_array = num_array[0:350000]
    num_list = num_array.tolist()
    feature_map = [([]) for i in range(categories)]
    ID_map = [([]) for i in range(categories)]
    for index,item in enumerate(num_list) :
        if  feature_map[item]:
            print item,index
            feature_map[item].append(feat_list[index])
            ID_map[item].append(index)
        else:
            feature_map[item].append(feat_list[index])
            ID_map[item].append(index)
    feature_map_cm = [() for i in range(categories)]
    for index,i in enumerate(feature_map):
        if i:
            print len(feature_map) 
            i = np.array(i)
            temp = cm.CUDAMatrix(i)
            feature_map_cm[index]=temp
    feature_array = np.array(feature_map_cm)
    ID_array = np.array(ID_map)
    return feature_map_cm,ID_array
if __name__ == '__main__':
    #1.put everything into memory:include net,feature,picture_path
    caffemodel='/home/scw4750/frelam_20161027/snapshot_iter_63390.caffemodel'
    deploy_file='/home/scw4750/frelam_20161027/deploy.prototxt'
    mean_file='/home/scw4750/frelam_20161027/mean.binaryproto'
    categories = 28
    path_npy = '/home/scw4750/frelam_20161027/path1_sp+zl.npy'
    feat_npy = '/home/scw4750/frelam_20161027/feat_sp+zl.npy'
    num_npy = '/home/scw4750/frelam_20161027/num_sp+zl.npy'
    #1.1load net
    net,transformer=loadnet(caffemodel,deploy_file,mean_file)
    #classify net
    caffemodel_cla='/home/scw4750/imagesdata/net/snapshot_iter_16380.caffemodel'
    deploy_file_cla='/home/scw4750/imagesdata/net/deploy.prototxt'
    mean_file_cla='/home/scw4750/imagesdata/net/mean.binaryproto'
    net_cla,transformer_cla=loadnet(caffemodel_cla,deploy_file_cla,mean_file_cla)
    #1.2load database
    feature_array,ID_array= loaddatabase(feat_npy,num_npy,categories)
    path_array = np.load(path_npy)
    #1.3transfer database to different table in memory for query
    #2.1after get the upload image
    image_files=['/home/scw4750/mac.jpg'] 
    cm.cuda_set_device (0)
    cm.cublas_init()
    #c = cm.CUDAMatrix(feature_zl_array)
    print 'load GPU already'
    #2.0add the socket code below here,recieve upload image
    try:
        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);
        print("create socket succ!")

        sock.bind(('10.24.2.47',12345))
        print('bind socket succ!')

        sock.listen(5)
        print('listen succ!')

    except:
        print("init socket error!")

#f=open("/home/scw4750/mac.jpg","wb")
    while True:
        print("listen for client...")
        
        conn,addr=sock.accept()
        f = open("/home/scw4750/mac.jpg","wb")
#print("get client")
#print(addr)
        print('rec connection')
        conn.settimeout(30)
        while True:
            szBuf=conn.recv(1024)
        #f.write('123')
            if szBuf:
                f.write(szBuf)  
                continue
            break       
        f.close()
        script_start_time = time.time()
        indices1,feat=classify(net,transformer,image_files)
        indices,feat1=classify(net_cla,transformer_cla,image_files)
        feat_norm=normolize(feat)
        QueryIDResult = search(indices,feat_norm,feature_array,ID_array)
        re = path_array[QueryIDResult[0:40]]
        
        #print re
        print 'Script took %f seconds.' % (time.time() - script_start_time,)
        print str(re)
        conn.sendall(str(re))
        
        #re = re[0:10]
        '''
        for image_dir in re:
            im = PIL.Image.open(image_dir[0])
            im.show()
        '''
        re = None
        conn.close()
        
    print("end of servive")



    
    #indices,feat=classify(net,transformer,image_files)
    #2.3feat_normolize
    #feat_norm=normolize(feat)
    #2.4search in the database in memory
    #QueryIDResult = search(indices,feat_norm,feature_array,ID_array)
    #print QueryIDResult
    #print path_array[QueryIDResult[0:50]]
    #print indices,feat
    

	

