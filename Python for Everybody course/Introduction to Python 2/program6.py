import urllib.request
import xml.etree.ElementTree as ET

#Checking if the link exists
def checkLink(url):
    try:
        urllib.request.urlopen(url).read()
        print('Checking if URL is correct')
        return True
    except:
        print("The link is not correct")
        return False

#Input and check link
while True:
    url =  input('Enter URL: ')
    if checkLink(url) == True:
        break
    else:
        continue

#Retrieving and parsing data from the site
data = urllib.request.urlopen(url).read()
print("Retieved", len(data), "characters")
tree = ET.fromstring(data)

#Making a dictionary with each name and its number, because I can
coolDict = dict()
names = tree.findall('.//name')
numbers = tree.findall('.//count')
for name,number in zip(names,numbers):
    #print(name.text ,number.text)
    coolDict[name.text] = number.text
#print(coolDict)
#HINT: to iterate through 2 items of 2 lists use zip

#Actually, lets make the sum though
countNames = 0
countNumbers = 0
for name,number in zip(names,numbers):
    countNames += 1
    countNumbers += int(number.text)
print("Count: ", countNames)
print("Sum: ", countNumbers)