#We want to know how many times emails have been sent at a certain hours
fhand = open("mbox-short.txt")
#print(fhand)
timeTimes = dict()
timeList = list()

for line in fhand:
    line = line.rstrip()
    words = line.split()
    if line.startswith("From "):
        #print(words)
        time = words[5]
        splitTime = time.split(":")
        #print(splitTime)
        hour = splitTime[0]
        #print(hour)
        timeList.append(hour)
#print(timeList)
#Now we have a list with each hour

#Next, we create a dictionary with each hour and the times it comes up
for time in timeList:
    timeTimes[time] = timeTimes.get(time, 0) + 1
#print(timeTimes)

#Last, we print it as the exercise wants
sortingHours = sorted(timeTimes.items())
#print(sortingHours)
for Time, Times in sortingHours:
    print(Time, Times)
