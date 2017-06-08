'''
Created on 2017. 4. 13.

@author: Byoungho Kang
'''
class AddLayer:
    
    def forward(self, x, y):
        return x+y
    
    def backward(self, difOut):
        dx = difOut
        dy = difOut
        return dx, dy

class MulLayer:
    
    # 순전파시 입력값 유지
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x*y
    
    def backward(self, difOut):
        dx = difOut*self.y
        dy = difOut*self.x
        return dx, dy
    
applePrice = 100
orangePrice = 150
appleCnt = 2
orangeCnt = 3
tax = 1.1

calAppleLayer = MulLayer()
calOrangeLayer = MulLayer()
calSumLayer = AddLayer()
calTaxLayer = MulLayer()

# forward Propagation
calApple = calAppleLayer.forward(applePrice, appleCnt)
calOrange = calOrangeLayer.forward(orangePrice, orangeCnt)
calSum = calSumLayer.forward(calApple, calOrange)
calTax = calTaxLayer.forward(calSum, tax)
print(calApple, calOrange, calSum, calTax)

# backward Propagation
difOut = 1
difCalSum, difCalTax = calTaxLayer.backward(difOut)
difCalApple, difCalOrange = calSumLayer.backward(difCalSum)
difApplePrice, difAppleCnt = calAppleLayer.backward(difCalApple)
difOrangePrice, difOrangeCnt = calOrangeLayer.backward(difCalOrange)
print(difOut, difCalSum, difCalTax, difCalApple, difCalOrange)
print(difApplePrice, difAppleCnt, difOrangePrice, difOrangeCnt)
