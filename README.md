# image_search_deep_learning
<div>
  <div>This project is base on caffe,cuda-python,socket.
  <div><br></div>
  <div>The pototype of&nbsp; patent and e-commerce comondity image search.This project is the retrieval part of the whole system</div>
  <div><br></div>
  <div>Retrieval code is in classify_main.py(classify_main2/3.py is for testing) ,including classification and retrieval code with CUDA accelerate code.</div>
  <div><br></div>
  <div>We use alexnet for feature extraction,with pretrain model on imagenet and finetune on our dateset</div>
  <div><br></div>
  <div>Here is our finetune model https://pan.baidu.com/s/1eSaq2oa with 93% top-3 accuracy on validation set.</div>
   <div><br></div>
  <div>We apply CUDA libreary-python-interface to accerate similarity computation,which allow us to retrieval in 0.2s for 300 thousand images at a single GPU </div>
<br></div>
  
  related paper“”
  
  http://yts.gdut.edu.cn/patentEarlyWarningSce/pa/pages/index.html (project page)

  performance can be improve in following:
  1. when loading the database into gpu， the memory comsume is high because features are all loaded into memory， wating to be transfer into gpu's memory. some improvement can be done to free the memory in time after features is transfered into gpu.
  
  2. think about the situation of dealing with several requesets for client. some mutil-thread technic should be use here.
  
  3. 
