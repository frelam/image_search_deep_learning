#!/usr/bin/env python2
# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files

Use this script as an example to build your own tool
"""

import argparse
import os
import time

from google.protobuf import text_format
import numpy as np
import PIL.Image
import scipy.misc

os.environ['GLOG_minloglevel'] = '2'  # Suppress most caffe output
import caffe  # noqa
from caffe.proto import caffe_pb2  # noqa
import cudamat as cm

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
        feat = net.blobs['fc6'].data
        end = time.time()
        if scores is None:
            scores = np.copy(feat)
        else:
            scores = np.vstack((scores, feat))
        print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    return scores


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


def classify(caffemodel, deploy_file, image_files,
             mean_file=None, labels_file=None, batch_size=None, use_gpu=True):
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
    net = get_net(caffemodel, deploy_file, use_gpu)
    transformer = get_transformer(deploy_file, mean_file)
    _, channels, height, width = transformer.inputs['data']
    if channels == 3:
        mode = 'RGB'
    elif channels == 1:
        mode = 'L'
    else:
        raise ValueError('Invalid number for channels: %s' % channels)
    images = [load_image(image_file, height, width, mode) for image_file in image_files]
    labels = read_labels(labels_file)

    # Classify the image
    feat = forward_pass(images, net, transformer, batch_size=batch_size)
    #normolize the feat
    '''
    cm.cuda_set_device (0)
    cm.cublas_init()
    a = cm.CUDAMatrix(feat)
    c = cm.dot(a, a.T)
    c =cm.sqrt(c)
    c = c.asarray()
    
    for index,item in enumerate(feat):
        feat[index,:]=item/(c[index][index])
    '''
    #save feature.npy
    f = file('/home/scw4750/frelam_20161027/get_feature/data/feature_5w-10w.npy', "wb")
    
    np.save(f, feat)
    #
    # Process the results
    #
    
    #indices = (-scores).argsort()[:, :5]  # take top 5 results
    #classifications = []
    #print feat[0:100,:]
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

if __name__ == '__main__':
    script_start_time = time.time()
    caffemodel='/home/scw4750/frelam_20161027/get_feature/python/20161107-150933-4ef1_epoch_30.0/snapshot_iter_6420.caffemodel'
    deploy_file='/home/scw4750/frelam_20161027/get_feature/python/20161107-150933-4ef1_epoch_30.0/deploy.prototxt'
    #train_txt_file='/home/scw4750/frelam_20161027/path.txt'
    #f2=open(train_txt_file)
    image_files=None
    image_files_temp = np.load('/home/scw4750/frelam_20161027/get_txt/path.npy')
    image_files_1 = image_files_temp.tolist()
    image_files = image_files_1[50000:100000]
    '''
    for line in f2:
        if image_files is None:
            image_files = [line.strip()]
        else:
            image_files.append(line.strip())
    '''
    #image_files=['mac.jpg']
    mean='/home/scw4750/frelam_20161027/get_feature/python/20161107-150933-4ef1_epoch_30.0/mean.binaryproto'
    classify(caffemodel,deploy_file,image_files,mean,None,256)
    print 'Script took %f seconds.' % (time.time() - script_start_time,)

	

