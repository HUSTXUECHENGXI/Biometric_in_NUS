#nus biometrics Assignment 1
# Part1
# Q1
# Add up the even numbers from 1 to 100 and output their sum, using while and for loops.(5)

def addevenfor():
	sum=0
	for a in range(1,101):
		if a%2!=0:
			continue
		else:
			sum=sum+a
	print(sum)


def addevenwhile():
	sum=0
	a=0
	while (a<100):
		a += 1
		if a%2!=0:
			continue
		else:
			sum=sum+a
	
	print(sum)


addevenfor()
addevenwhile()