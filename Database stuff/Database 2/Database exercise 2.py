import sqlite3

conn = sqlite3.connect('emaildb.sqlite')
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS Counts')

cur.execute("CREATE TABLE Counts (org TEXT, count INTEGER)")

#input the name file and check if it's the correct one
while True:
    fname = input('Enter file name: ')
    if fname == "mbox.txt":
        break
    else :
        print("File name is not correct.")
fh = open(fname)

#We make a list of domains, to have a clear set of data we will need
domainList = list()

#At first, we search for email addresses with a @, then we cut off everything apart the domain (org)
for line in fh:
    if line.startswith("From "):
        words = line.split()
        email = words[1]
        #print(email)
        domain = email[email.find("@")+1:]
        #print(domain)
        domainList.append(domain)
#print(domainList)

#Now, all domains are clear and in order in our list, we then add them to our table

for item in domainList:
    cur.execute('SELECT count FROM Counts WHERE org = ? ', (item,))
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (org, count) VALUES (?, 1)''', (item,))
    else:
        cur.execute('UPDATE Counts SET count = count + 1 WHERE org = ?', (item,))
conn.commit()

#Now that we have created out "virtual" database,
# https://www.sqlite.org/lang_select.html
sqlstr = 'SELECT org, count FROM Counts ORDER BY count DESC LIMIT 10'

for row in cur.execute(sqlstr):
    print(str(row[0]), row[1])

cur.close()
