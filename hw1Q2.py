# nus biometrics Assignment 1
# Part1
# Q2
# Read a string from console and output its length, convert it to lower case and upper case, and reverse it. (Hint: try string slice with step -1) (5)

def Q2():
	s=input("please enter your string:")
	lens=len(s)
	print(lens)
	print (s.lower())
	print (s.upper())
	le=s[::-1]
	print(le)
	
Q2()