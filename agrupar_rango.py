#import utils
import numpy as np
#array = [902,950,1400,1401,1423,1446,1501,7000]
#array = [1,3,9,11,180,181,200]
array = [1,5,15,20]
# array=[1]

array.sort()
array_diff = np.ediff1d(array)

curr_cnt=0
max_limit=5
range_found=False
sol=[]
# for d,i in enumerate(array_diff):
# 	if not range_found:
# 		if d<max_limit:
# 			range_found=True
# 			interval=[array[i],array[i+1]]
# 			curr_cnt+=d
# 		else:
# 			sol.append(array[i])
# 	else:
# 		if curr_cnt+d<max_limit:
# 			interval.append(array[i+1])
# 			curr_cnt+=d
# 		else:
# 			curr_cnt=0
# 			chosen_one=interval[len(interval)/2]
# 			range_found=False
# 			sol.append(chosen_one)

for i,a in enumerate(array[:-1]):
	d=array_diff[i]

	if d>max_limit:
		if not range_found:
			sol.append(a)
		else:
			curr_cnt=0
			chosen_one=interval[len(interval)/2]
			range_found=False
			sol.append(interval)

	else:
		if not range_found:
			range_found=True
			interval=[array[i],array[i+1]]
			curr_cnt+=d
		else:
			if curr_cnt+d<max_limit:
				interval.append(array[i+1])
				curr_cnt+=d 
			else:
				curr_cnt=0
				chosen_one=interval[len(interval)/2]
				range_found=False
				sol.append(interval)

if not range_found:
	sol.append(array[-1])
else:
	sol.append(interval)

print sol


