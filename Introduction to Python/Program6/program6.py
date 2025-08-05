fileInput = input("Enter file name: ")
fName = open(fileInput)

#initiaite sum and count to then make the average
summing = 0
count = 0

#read the text, extract the float, make up sum and count to the compute the average
for line in fName:
    if line.startswith("X-DSPAM-Confidence:"):
        print(line.rstrip())
        doubleDotPosition = line.find(":")
        print(doubleDotPosition)
        summing = summing + float(line[doubleDotPosition + 2:])
        print(sum)
        count = count + 1
        print(count)
        print("")

#Lastly the average
average = summing / count
print("Average spam confidence:", average)