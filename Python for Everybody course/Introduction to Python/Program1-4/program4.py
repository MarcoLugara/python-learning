text = "X-DSPAM-Confidence:    0.8475"

startPosition = text.find("0")
print(startPosition)
endPosition = text.find('5',startPosition)
print(endPosition)

rawGoal = text[startPosition : endPosition+1]
goal = float(rawGoal)
print(goal)
