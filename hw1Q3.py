# nus biometrics Assignment 1
# Part1
# Q3
# Read a string from console. Split the string on space delimiter (” ”) and join using a hyphen (”-”). (Example: input the string ”this is a string” and output as ”this-is-astring”) (5)
def Q3():
	s=input("please enter your string:")
	s1=s.replace(' ','-')
	print(s1)

Q3()