'''
Created on 2017. 4. 5.

@author: Byoungho Kang
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

# 데이터준비
x = np.arange(0, 6, 0.1) 
y1 = np.sin(x)
y2 = np.cos(x)
print(x)
print(y1)
print(y2)

# 그래프 그리기
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()

# 이미지 표시하기
img = image.imread("../resources/Scream.jpg")
plt.imshow(img)
plt.show()