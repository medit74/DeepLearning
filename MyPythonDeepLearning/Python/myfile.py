'''
Created on 2017. 4. 7.

@author: Byoungho Kang
'''

'''
파일열기 
open(filename, mode)
r-read
w-write
a-append
b-binary
'''
f = open("../resources/newfile.txt","w")
for idx in range(1, 6):
    data = "%d번째 줄입니다.\n" % idx
    f.write(data)
f.close()

f = open("../resources/newfile.txt","r")
data = f.read()
print(data)
f.close()

'''
with문을 이용해서 파일열기
with block을 벗어나면 파일객체는 자동 Close
'''

with open("../resources/newfile.txt","r") as f:
    f.read()
    print(data)
    
