#-*- coding: UTF-8 -*-
import re
import os
import os.path

folder_path = '/home/scw4750/frelam_20161027/path.txt'
director_path = '/home/scw4750/frelam_20161027/path_1.txt'

f = open(folder_path, "r")
fw = open(director_path,"w")
while True:
    line = f.readline()
    if line:
        lines=line.rsplit('/',1)
        fw.write(lines[1])
        #lines[1]
        #print lines[1]
    else:
        break
fw.close()
f.close()
