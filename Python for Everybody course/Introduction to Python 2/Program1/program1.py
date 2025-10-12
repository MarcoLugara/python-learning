import re

fhand = open("text2.txt")
text = fhand.read()
#print(fhand)

#Finding all numbers in the file and making a list with all of them
numbersList = re.findall("[0-9]+", text)
#print(numbersList)

#Sum computation
Sum = 0
for number in numbersList:
    Sum = Sum + int(number)
    #print(Sum)
print(Sum)
