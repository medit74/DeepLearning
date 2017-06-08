'''
Created on 2017. 4. 7.

@author: Byoungho Kang
'''

'''
import (패키지이름).모듈이름 
from (패키지이름).모듈이름 import 모듈함수
'''
import EnvironmentSetup.myfunction as mf
from EnvironmentSetup.myfunction import noReturn
 
print("-- mymodule.py --")
print(mf.sayHello("Tiger"))
noReturn()