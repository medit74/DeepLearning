'''
Created on 2017. 4. 7.

@author: Byoungho Kang
'''

import time

localtime = time.localtime(time.time())
localhour = localtime.tm_hour
print(localtime, type(localtime))
print(localtime.tm_hour, type(localtime.tm_hour))

#if statement
print("\n-- if statement --")
greeting = ""
if(localhour >= 6 and localhour < 11):
    greeting = "Good Morning"
elif(localhour >= 11 and localhour < 17):
    greeting = "Good Afternoon"
elif(localhour >= 17 and localhour <22):
    greeting = "Good Evening"
else:
    greeting = "Hello"
print(greeting, type(greeting))

#while statement
print("\n-- while statement --")
count = 0
while(count < 9):
    print ('The count is: ', count, type(count))
    count = count + 1

#for statement
print("\n-- for statement --")
fruits = ['banana','apple','mango']
for item in fruits:
    print (item, type(item))
for idx in range(len(fruits)):
    print (idx, "-", fruits[idx])