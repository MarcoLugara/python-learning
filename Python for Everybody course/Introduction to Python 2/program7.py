import urllib.request
import json

#Checking if the link exists
def checkLink(url):
    try:
        urllib.request.urlopen(url).read()
        #print('Checking if URL is correct')
        return True
    except:
        print("The link is not correct")
        return False

#Input and check link
while True:
    url =  input('Enter location: ')
    if checkLink(url) == True:
        print("Retrieving: ", url)
        break
    else:
        continue

#Retrieving and parsing data from the site
data = urllib.request.urlopen(url).read()
print("Retieved", len(data), "characters")
data = json.loads(data)
#print(data)

dataList = data["comments"]
# print(type(dataList)) è una lista di dizionari

#Create a list of names and a list of numbers
namesList = list()
numbersList = list()
#print(namesList)
#print(numbersList)

for item in dataList:
    #print(item["name"])
    #print(item["count"])
    namesList.append(item["name"])
    numbersList.append(int(item["count"]))
#print(namesList)
#print(numbersList)

#Sum of numbers and count of people
Count = 0
totalCount = 0
for number in numbersList:
    Count += 1
    totalCount += number
print(Count)
print(totalCount)