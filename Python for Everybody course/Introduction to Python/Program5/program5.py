#prompting the file
fname = input("Enter file name: ")

fhand = open(fname)
rawText = fhand.read()
text = rawText.rstrip()
TEXT = text.upper()
print(TEXT)