#Opening and dividing the file line by line
fhand = open("romeo.txt")
#print(fhand)

#The empty list we are going to fill up
goal = list()

#Splitting each line in a list of words
for line in fhand:
    #We create the raw unordered list we want
    words = line.split()
    #print(words)
    for word in words:
        goal.append(word)
    #print(goal)

trueGoal = list()
#We remove double words
for word in goal:
    #print(word)
    if word not in trueGoal:
        trueGoal.append(word)
        #print(trueGoal)
    else:
        continue

#Now, we reorder our list
trueGoal.sort()
print(trueGoal)