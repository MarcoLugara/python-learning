#Searching who sent the most mails
fhand = open("mbox-short.txt")
counts = dict()

#Making a list with each email address sending messages and how many times they sent a message (histogram)
for line in fhand:
    line = line.rstrip()
    if line.startswith("From "):
        words = line.split()
        #print(words)
        email = words[1]
        counts[email] = counts.get(email, 0) + 1
#print(counts)

#Finding the email address that sent the most
bigEmail = None
bigCount = None
for email, count in counts.items():
    if bigEmail == None or count > bigCount:
        bigEmail = email
        bigCount = count

print(bigEmail, bigCount)