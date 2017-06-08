'''
Created on 2017. 4. 7.

@author: Byoungho Kang
'''
# Number
print(123)
print(1.23)
print(1 + 2j) #복소수
print(0o34) #8진수
print(0xFF) #16진수

# String
print('Hello')
print("What's your name?")
print('"My name is Tiger" he said')
print("Life is too short.\nYou need python.")
print("I've been to %d countries." % 15)
print("%s is the best country in %d" % ("Swiss", 15))

# list type 
list1 = ['my','python','list']
print(list1, type(list1))
list1.append('added')
print(list1)

# tuple type (the tuples cannot be changed unlike lists)
tuple1 = ('my','python','tuple')
print(tuple1, type(tuple1))
tuple2 = 'this','is','tuple','type','too'
print(tuple2, type(tuple2))

# dictionary type (Each key is separated from its value by a colon (:))
dict1 = {'company':'SKC&C','location':'Seongnam'}
print(dict1, type(dict1))
dict1['company'] = 'SK'
dict1['slogan'] = 'Creative ICT Factory'
print(dict1, type(dict1))

# set type 
set1 = set([1,1,3,3,5,5])
print(set1, type(set1))
set2 = set("Hello")
print(set2, type(set2))
