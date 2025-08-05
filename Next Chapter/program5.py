#Extracting clickable links from a webpage
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

#Define a function: starting from a link, we read the web page and find the wanted link
def openLinkFindLink(url):
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    #print(soup)

    # Finding the clickable links
    tagsTest = soup("a")  # is equal to tagsTest = soup.find_all("a"),
    # i.e. finding all the <a> .... </a> and printing them on a list
    # print(tagsTest)
    # print(type(tagsTest))
    # For each <a> .... </a>, we get only the hyperlink, i.e. the href and put them in a list
    linksList = list()
    for line in tagsTest:
        link = line.get("href")
        # print(link)
        # print(type(link)) returns a string
        linksList.append(link)
    # print(linksList)
    try:
        url = linksList[position - 1]
        print(url)
    except:
        url = False
    return url

#Checking if the input for Count and position are correct and making them integers
def checkCount(c):
    #Count check
    try:
        c = int(c)
    except:
        print("Count must be an integer")
        return False
    if int(c) <= 0:
        print("Invalid input, print a correct number for the count")
        return False
    else:
        return True
def checkPosition(p):
    try:
        p = int(p)
    except:
        print("Position must be an integer")
        return False
    if int(p) <= 0:
        print("Invalid input, print a correct number for the position")
        return False
    else:
        return True

#Checking if the link exists
def checkLink(site):
    try:
        linkOpening = urllib.request.urlopen(site).read()
        return True
    except:
        print("The link is not correct")
        return False

#Testing if I can make this whole process as a function
# with the "str" type url that the function would return
#print(type(chosenLink))
#html2 = urllib.request.urlopen(chosenLink).read()
#soup2 = BeautifulSoup(html, "html.parser")
#print(soup2)

#Allowing the user to choose the starting link, the
# position of the link to open and how many times to repeat the process
# Link to use is http://py4e-data.dr-chuck.net/known_by_Ayda.html

#Input and check link
while True:
    url =  input('Enter URL: ')
    if checkLink(url) == True:
        break
    else:
        continue
#Input and check count
while True:
    Count =  input('Enter count: ')
    if checkCount(Count) == True:
        Count = int(Count)
        break
    else:
        continue
#Input and check position
while True:
    position =  input('Enter position: ')
    if checkPosition(position) == True:
        position = int(position)
        break
    else:
        continue

#testCount = 7
#testPosition = 18

#Iteration test
#secondLink = openLinkFindLink(url)
#print(secondLink)

#Building the iteration with the user-chosen count and the user chosen position
while Count > 0:
        Count -= 1
        if openLinkFindLink(url) == False:
            print("Position number is too high!")
            while True:
                position = input('Enter position: ')
                if checkPosition(position) == True:
                    position = int(position)
                    break
                else:
                    continue
        url = openLinkFindLink(url)


