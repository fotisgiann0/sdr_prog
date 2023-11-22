import numpy as np

ksiedge = 4
m = 3
n = 10
q = m*n + n + 1
q2 = m*n + n + 2
q3 = m*n + n
q4 = m*n + n + 3
p = m*n + n #dokimastiki timi
h = m -1  #dokimastiki timi
jj_row = 2 #dokimastiki timi
ptx = 85
prx = 99
le = 7
lt = 8



C1 = np.zeros([q,q])
#a0 = np.ones([q,1])
a1 = np.zeros([q,1])
#a2 = np.ones([1,q])
a3 = np.zeros([1,q])
a4 = np.zeros([2,1])
a5 = np.zeros([1,2])
a6 = np.zeros([2,2])
aq2 = np.zeros([q2,q2])



# gia ton A0 


w = np.empty((1,n), float)
for i in range(n):
    w[0][i] = i
print(w)

#li = a0, g1 einai o adj tou a0

g0 = []
g0.append(w)
g0.append(np.zeros([1,q - n]))
g1 = np.block(
    g0
)
print(g1)
a00 = np.empty((q,1), float)
for i in range(q):
    a00[i][0] = g1[0][i]

A0=[]
row1 =[]
row2 = []
row3 = []
fini = []
row1.append(C1)
row1.append(a00)
row1.append(a1)
fini.append(row1)
row2.append(g1)
row2.append(a5)
fini.append(row2)
row3.append(a3)
row3.append(a5)
fini.append(row3)

A0 = np.block(
    fini
)
print(A0)
# a2 einai o a2, g3 einai o adj tou a2
g2 = []
g2.append(np.zeros([1,q2 - 2]))
g2.append(-1)
g2.append(0)
g3 = np.block(
    g2
)

a2 = np.empty((q2,1), float)
for i in range(q2):
    a2[i][0] = g3[0][i]
print(a2)

#b2 einai o b2, g5 einai o adj tou b2
g4 =[]
g4.append(w)
g4.append(np.zeros([1,q2 - n + 1]))
g5 = np.block(
    g4
)
b2 = np.empty((q2+1,1), float)
for i in range(q2+1):
    b2[i][0] = g5[0][i]
print(b2)

A2 = []
a21 =[]
r1 =[]
r2 = []
r1.append(aq2)
r1.append(0.5 * a2)
a21.append(r1)
r2.append(0.5 * g3)
r2.append(0)
a21.append(r2)
A2 = np.block(
    a21
)
print(A2)


a31 = []
for i in range(m + 1):
    a31.append(np.identity(n))
a31.append(np.zeros([n,3]))
A3 = np.block(
    a31
)
print(A3)

ro1 = []
ro2 = []
ro3 = []
a41 = []
A4 =[]
ro1.append(np.zeros([q3,q3]))
ro1.append(np.zeros([q3,1]))
ro1.append(np.zeros([q3,2]))
a41.append(ro1)
ro2.append(np.zeros([1,q3]))
ro2.append(1)
ro2.append(np.zeros([1,2]))
a41.append(ro2)
ro3.append(np.zeros([2,q3]))
ro3.append(np.zeros([2,1]))
ro3.append(np.zeros([2,2]))
a41.append(ro3)
A4 = np.block(
    a41
)
print(A4)

#k4 einai o adj tou b4
b41 = []
k4 = []
b41.append(np.zeros([1,q]))
b41.append(-1)
b41.append(0)
k4 = np.block(
    b41
)
b4 = np.empty((q+2,1), float)
for i in range(q+2):
    b4[i][0] = k4[0][i]
print(b4)



b51 = []
k5 = []
b51.append(np.zeros([1,q3]))
b51.append(-1)
b51.append(np.zeros([1,2]))
k5 = np.block(
    b51
)
b5 = np.empty((q3+3,1), float)
for i in range(q3+3):
    b5[i][0] = k5[0][i]
print(b5)


dul = np.empty((n,m), float)
ddl = np.empty((n,m), float)
Dk = np.empty((n,m), float)

#dinw arxikes times stous pinakes dul kai ddl kai Dk gia testing
for i in range(n):
    for j in range(m):
        dul[i][j] = 4
        ddl[i][j] = 5
        Dk[i][j] = 6



#edw 3ekinaei to ktisimo tou b0',o opios  einai o pt3
pt1 = []
pt2 = []

for i in range(m):
    for j in range(n):
        pt1.append(ptx * dul[j][i])
        pt2.append(prx * ddl[j][i])

#print(pt1)
pt11 = np.empty((n*m,1), float)
for i in range(n*m):
    pt11[i][0] = pt1[i]
#print(pt11)
pt22 = np.empty((n*m,1), float)
for i in range(n*m):
    pt22[i][0] = pt2[i]
#print(pt22)


#pt3 einai o b0'
pt3 = np.empty((n*m,1), float)
for i in range(n*m):
    pt3[i][0] = pt11[i][0] + pt22[i][0]
print("edw einai o bo'", pt3)

#gia ton b0
b0 = np.empty((q+2,1), float)
for i in range(n):
    b0[i][0] = 0
for i in range(n*m):
    b0[i+n][0] = le * pt3[i][0]
b0[n+n*m] = 0
b0[n+n*m + 1] = 0
b0[n+n*m + 2] = lt 

print("edw einai o b0", b0)

#gia ton A1
A111 = []
A11 = []
A1 = []
for i in range(m):
    A111.append(np.zeros([1,n]))
    for j in range(i):
        A111.append(np.zeros([1,n]))
    A111.append(Dk[i])
    for coun in range(m-i):
        A111.append(np.zeros([1,n]))
    A111.append(0)
    A111.append(0)
    A111.append(-1)
   # print(A111)
   #A11 = np.hstack((A111))
    A11.append(A111)
  #  print(A11)
    A111 = []
# print(A11)
# print(Dk[i])
A1 = np.block(
        A11
    )
print(A1)

#arithmisi apo to 0 sto up
def create_up(sp):
    up = np.zeros([n*m + n + 3, 1])
    up[sp-1][0] = 1
    #print(up)
    return up


def diag_up(sp):
    up = np.zeros([n*m + n + 3, n*m + n + 3])
    up[sp-1][sp-1] = 1
    return up


# x = create_up(4)
# print(diag_up(4))


#SDR arrays
B00 =[]
first_row = []
second_row = []
final_array = []
#b0t einai o adj tou b0
b0t = np.empty((1, q+2), float)
for i in range(q+2):
    b0t[0][i] = b0[i][0]



first_row.append(A0)
print("edw first row " ,first_row)
first_row.append(0.5*b0)
final_array.append(first_row)
print("edw final arr", final_array)
second_row.append(0.5*b0t)
print("edw second row " ,second_row)
second_row.append(0)
print("edw second row " ,second_row)
final_array.append(second_row)
print("edw final arr", final_array)
B00 = np.block(
    final_array
)
print("edw eibai o B0", B00)


#b2 einai o b2, g5 einai o adj tou b2
B20 = []
first_row = []
second_row = []
final_array = []
#print("first, sec,...", first_row, second_row, final_array)

first_row.append(A2)
first_row.append(0.5*b2)
final_array.append(first_row)
second_row.append(0.5*g5)
second_row.append(0)
final_array.append(second_row)
B20 = np.block(
    final_array
)
print("edw einai o B2", B20)

#k4 einai o adj tou b4
B40 = []
first_row = []
second_row = []
final_array = []

first_row.append(A4)
first_row.append(0.5*b4)
final_array.append(first_row)
second_row.append(0.5*k4)
second_row.append(0)
final_array.append(second_row)
B40 = np.block(
    final_array
)
print("edw einai o B4", B40)

#k5 einai o adj tou b5
B50 = []
first_row = []
second_row = []
final_array = []

first_row.append(np.zeros([q4,q4]))
first_row.append(0.5*b5)
final_array.append(first_row)
second_row.append(0.5*k5)
second_row.append(0)
final_array.append(second_row)
B50 = np.block(
    final_array
)
print("edw einai o B5", B50)

#Gp 
Gp_ol = []
for j in range(m*n + n):
    Gp = []
    first_row = []
    second_row = []
    final_array = []

    up_adj = np.zeros([1, n*m + n + 3])
    up_adj[0][j] = 1
    first_row.append(diag_up(j))
    first_row.append(-0.5*create_up(j))
    final_array.append(first_row)
    second_row.append(-0.5*up_adj)
    second_row.append(0)
    final_array.append(second_row)
    Gp = np.block(
        final_array
    )
    Gp_ol.append(Gp)
#     print("edw einai o Gp", Gp)
#print("edw einai o Gp", Gp_ol[m*n + n - 1] == Gp)

#Hh
Hh_ol = []
for j in range(m):
    Hh = []
    first_row = []
    second_row = []
    final_array = []
    hh1 = np.zeros([n+ m*n + 3 ,1])  #grammi h tou A1 kai adj tis grammis
    hh2 = np.zeros([1,n+ m*n + 3])
    for i in range(n+ m*n + 3):
        hh1[i][0] = A1[j][i]
        hh2[0][i] = A1[j][i]


    first_row.append(np.zeros([q4,q4]))
    first_row.append(0.5*hh1)
    final_array.append(first_row)
    second_row.append(0.5*hh2)
    second_row.append(0)
    final_array.append(second_row)
    Hh = np.block(
    final_array
    )
    #print("edw einai o Hh", Hh)
    Hh_ol.append(Hh)
#print("edw einai o olikos", Hh_ol[0])

#Jj
Jj_ol = []
for j in range(n):
    Jj = []
    first_row = []

    second_row = []
    final_array = []
    jj1 = np.zeros([m*n + 3 + n,1])  #grammi jj_row tou A3 kai adj tis grammis
    jj2 = np.zeros([1,m*n + n + 3])
    for i in range(m*n + 3 + n):
        jj1[i][0] = A3[j][i]
        jj2[0][i] = A3[j][i]


    first_row.append(np.zeros([q4,q4]))
    first_row.append(0.5*jj1)
    final_array.append(first_row)
    second_row.append(0.5*jj2)
    second_row.append(0)
    final_array.append(second_row)
    Jj = np.block(
        final_array
    )
    Jj_ol.append(Jj)
#     print("edw einai o Jj", Jj)
# print("edw einai o Jj", Jj_ol[2])

print("B0 size", len(B00))