'''
Created on 2017. 4. 12.

@author: Byoungho Kang
'''

class Calculator:

    def __init__(self, first, second):
        self.first = first
        self.second = second    
        
    def plus(self):
        return self.first + self.second
    
    def minus(self):
        return self.first - self.second
    
    def multiply(self):
        return self.first * self.second 
    
    def divide(self):
        return self.first / self.second


c = Calculator(3, 4)
print(c.plus(), c.minus(), c.multiply(), c.divide())