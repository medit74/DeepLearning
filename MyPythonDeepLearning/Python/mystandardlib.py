'''
Created on 2017. 4. 7.

@author: Byoungho Kang
'''

import sys
import os
import pickle
import time
import calendar
import random

# sys - Path 경로
print(sys.path)

# os- Directory
print(os.pardir)
print(os.getcwd())

# pickle - 객체의 형태를 유지하면서 파일에 저장하고 불러올수 있게 하는 객체
f = open("../resources/pickletest.txt","wb")
data = {'company':'SK','location':'Seongnam','slogan':'Creative ICT Factory'}
pickle.dump(data,f)
f.close
f = open("../resources/pickletest.txt", "rb")
data = pickle.load(f)
print(data)

# time - Date and time
print(time.time())
print(time.localtime(time.time()))
print(time.strftime('%Y.%m.%d %X', time.localtime(time.time())))

# calendar
localtime = time.localtime(time.time())
print(calendar.prmonth(localtime.tm_year, localtime.tm_mon))

# random
print(random.random())
print(random.randint(1,100))
data = [1,2,3,4,5]
random.shuffle(data)
print(data)