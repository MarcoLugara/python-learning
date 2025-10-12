import urllib.request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

#Open and read through the document
url = input("Enter - ")
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, "html.parser")

#NOTE TO SELF: regex work only with strings, in this case we have a tag, as type of soup, so regex methods dont work
Sum = 0
count = 0
#print(soup)
spans = soup.find_all("span")
#In sites there is the start of a thing with <name> lalalala </name>.
# To recall that piece of information, u use find_all with ("name") as the argument and Python
#  will return a list with <name> lalalala </name>, for all the pieces with that tag in the web page
#print(spans)
for span in spans:
    count += 1
    text = span.get_text() #or text = span.text() this would extract text without tags in web pages
    #print(text)
    Sum = Sum + int(text)
print("Count: ",  count)
print("Sum: ", Sum)
