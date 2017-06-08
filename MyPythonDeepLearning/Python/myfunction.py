'''
Created on 2017. 4. 7.

@author: Byoungho Kang
'''

def sayHello(name):
    greeting = "Hello " + name + "."
    return greeting

def noReturn():
    print("This function doesn't have return.")

def keywordArgs(name, age):
    print("Hello! I'm", name + ",", age, "years old.")
    
def defaultArgs(name, age, nationality="korean"):
    print("Hello! I'm", name + ",", age, "years old. I'm a", nationality)    
    
def variableLengthArgs(sentence, *args):
    for idx in range(len(args)):
        if (idx == 0):
            sentence = sentence + " " + args[idx]
        elif (idx <= len(args)-2):
            sentence = sentence + ", " + args[idx]
        else:
            sentence = sentence + " and " + args[idx] + "."
    print(sentence)

def returnTuples():
    cust1 = "Cass"
    cust2 = "Singha"
    return cust1, cust2

if __name__ == "__main__":
    print(sayHello('Singha'))
    noReturn()
    keywordArgs(age="44", name="Singha")
    defaultArgs("Cass", "23")
    variableLengthArgs("My hobbies are", "soccer", "horse riding", "travelling")
    variableLengthArgs("I've been to", "China", "Vietnam", "U.S.A", "India")
    customers = returnTuples()
    (cust1, cust2) = returnTuples()
    print(customers, type(customers))
    print(cust1, type(cust1))