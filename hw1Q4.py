# nus biometrics Assignment 1
# Part1
# Q4
# Learn the Python list operations and follow the commands below: (5) 
#• Initialize an empty list L. 
#• Add 5, 10, 3 to the list. 
#• Insert 9 to the head of the list; 
#• Double the list. (e.g. change L = [1,2,3] to L = [1,2,3,1,2,3])
#• Remove all 10 in the list. 
#• Reverse the list.

def Q4():
	L=[]
	print(L)
	L.append(5)
	print(L)
	L.append(10)
	print(L)
	L.append(3)
	print(L)
	L.insert(0,9)
	print(L)
	L=L+L
	print(L)
	def filter1(x):
		if x==10:
			return False
		return True
	L=list(filter(filter1,L))
	#for method can also solve this problem ,like in the class
	print(L)
	L.reverse()
	print(L)

Q4()
