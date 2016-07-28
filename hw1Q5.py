# nus biometrics Assignment 1
# Part1
# Q5
#5. Deﬁne a function for insertion sort and use the code below to test you sort function. (10)
#import random
#def InsertSort ( seq ): #define your function here
#testseq = [ ] for i in range (20): testseq . append(random. randint (1 , 200)) print ( testseq ) print ( InsertSort ( testseq ))
import random

def InsertSort (seq):
	for i in range(1, len(seq)):
		s = seq[i]
		j = i
		while j > 0 and seq[j - 1] > s:
			seq[j] = seq[j - 1]
			j -= 1
			seq[j] = s
	return seq
#define your function here
testseq = [ ] 
for i in range (20): 
	testseq . append(random. randint (1 , 200)) 
print ( testseq ) 
print ( InsertSort (testseq))
