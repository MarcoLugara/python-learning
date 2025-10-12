smallest = None
biggest = None
tries = None

#finding the smallest, function
def FindSmall(s, t):
    if s is None:
        s = t
        #print(s)
    elif t < s:
        s = t
        #print(s)
    else:
        s = s
    #print(s)
    return s

#finding the biggest, function
def FindBig(b, t):
    if b is None:
        b = t
    elif b < t:
        b = t
    else:
        b = b
    return b

while True:
    rawTries = input("Enter a number: ")
    if rawTries == "done":
        print("Maximum is", biggest)
        print("Smallest is", smallest)
        break
    try:
        tries = int(rawTries)
        biggest = FindBig(biggest, tries)
        print(biggest)
        smallest = FindSmall(smallest, tries)
        print(smallest)
    except:
        print("Invalid input")
