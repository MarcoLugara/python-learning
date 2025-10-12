#Extracting clickable links from a webpage
import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup
import re

#Reading the webpage
url = input('Enter URL: ')
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, "html.parser")
#type(soup)
#print(soup)

#Extracting the clickable links
finalList = list()
tags = soup("a")
#print(tags)
for tag in tags:
    rawTryLinks = tag.get("href", None)  #"href" finds the clickable links in a web page
    #None means that it will return None if there is nothing
    links = re.findall("https?://.+", rawTryLinks)
    for link in links:
        if link != None:
            finalList.append(link)
print(finalList)