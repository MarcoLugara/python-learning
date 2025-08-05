#Opening the file
fhand = open("mbox-short.txt")
#check_text = fhand.read()
#print(check_text)
count = 0

#Dividing it line by line and taking only the lines starting with "From",
# (extra: then taking only the piece of the email address after @)
for line in fhand:
    #print(line)
    if line.startswith("From "):
        count = count + 1
        #print(count)
        words = line.split()
        #print(words)
        email = words[1]
        print(email)

#Last sentence
print("There were", count, "lines in the file with From as the first word")

