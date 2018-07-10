
dir_path = '/home/scw4750/frelam_20161027/get_txt/txt/train_shangpin.txt'
direct_path = '/home/scw4750/frelam_20161027/get_txt/txt/path_shangpin.txt'
number_path = '/home/scw4750/frelam_20161027/get_txt/txt/number_shangpin.txt'


f1 = open(dir_path)
w1 = open(direct_path,'w+')
w2 = open(number_path,'w+')


for line in f1:
    line = line.replace('\n','')
    index1 = line.find(' ',0)
    direct_line = line[:index1]
    number_line = line[index1+1:]
    w1.write(direct_line + '\n')
    w2.write(number_line + '\n')

f1.close()
w1.close()
w2.close()
