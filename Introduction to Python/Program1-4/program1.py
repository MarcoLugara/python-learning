score = input("Enter Score: ")
nScore = float(score)

if nScore > 1:
    print("error")
elif nScore >= 0.9:
    print("A")
elif nScore >= 0.8:
    print("B")
elif nScore >= 0.7:
    print("C")
elif nScore >= 0.6:
    print("D")
elif nScore >= 0:
    print("F")
else:
    print("error")