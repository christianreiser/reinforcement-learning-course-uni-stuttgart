import numpy as np


def generate_list_of_all_policies(start,end,base,step=1):

    def Convert(n,base):
       string = "0123456789"
       if n < base:
          return string[n]
       else:
          return Convert(n//base,base) + string[n%base]
    return (Convert(i,base) for i in range(start,end,step))


all_policies = list(generate_list_of_all_policies(0,4**3,4))
print(len(all_policies))
for i in range(0, 4**9):
    a = list(map(int, [int for int in all_policies[i]]))
    #print('a=',a)
    b = np.zeros(9, dtype=np.int)
    for ele in range(0, len(a)):
        b[len(b)-ele-1] = a[len(a) - ele-1]
    print('b=',b)



