#!/usr/bin/env python3
import cvxpy as cp
import numpy as np
import itertools
import random
import sys
from math import exp, log
np.set_printoptions(threshold=sys.maxsize)
###################### basic settings
m = 2
n = 10  
p = n*m + n 
q = m*n + n + 1
q2 = m*n + n + 2
q3 = m*n + n
q4 = m*n + n + 3
iterations = 150
np.random.seed(5)

###################### constants
# d = np.random.uniform(1,2,size=(n,))
# s_initial = np.random.uniform(0.1,1,size=(n,))
# # b = np.random.uniform(15,95,size=(m,))
# R = np.random.uniform(5,10,size=(n,m))
# bmax = np.random.uniform(n/4,n/3,size=(m,))
# # print (bmax)
# Lmax = 0.9 # 0.3 #seconds
# Amin = 0.7 #0.7 
# zlocal = 0.011
# ksilocal = 0.7676
# zedge = 0.02862
# ksiedge = 0.06706
# alpha = 1 # change from 0.1 to 500 
#-----------------------------------
ptx = 1.285
prx = 1.181
le = 0.5
lt = 1 - le
L = 100
s_elliniko = 150 
rmin = 400 #* (10**3) #(400-s_elliniko) #* (10**6) 
rmax = 800 #* (10**3) #(400+s_elliniko) #* (10**6)
m_elliniko = 10
p_elliniko = 1.25 * (10**(-8))  
z_elliniko = 3
Ck_ul = [0.5, 0.5, 0.5]#np.random.uniform(10,20,size=(m+1,)) 
Ck_dl = [0.2, 0.1, 0.5]#np.random.uniform(10,20,size=(m+1,)) 

ai = np.empty((n), float) 
for i in range(n):
    ai[i] =  0.5 #(0.1 + i*0.1) 

w = np.empty((1,n), float)
for i in range(n):
    w[0][i] = 330 * ai[i] 

bi = np.empty((n), float)
for i in range(n):
    bi[i] = 0.2 * ai[i]
r_k = np.empty((m+1), float)
for i in range(m+1):
    r_k[i] = 0 
r_k[0] = 400 #400 * (10**6)
r_k[1] = 2000#2 * (10**9)
r_k[2] = 2200#2.2 * (10**9)
dul = np.empty((n,m+1), float)
ddl = np.empty((n,m+1), float)
Dk = np.empty((n,m+1), float)

for i in range(n):
    for j in range(m+1):
        if(j == 0):
            dul[i][0] = 0
            ddl[i][0] = 0
        else :
            dul[i][j] = ai[i] / Ck_ul[j] 
            ddl[i][j] = bi[i] / Ck_dl[j] 
        Dk[i][j] = dul[i][j] + ddl[i][j] + (w[0][i] / r_k[j])



#arithmisi apo to 0 sto up
def create_up(sp):
    up = np.zeros([n*m + n + 3, 1])
    up[sp][0] = 1
    #print(up)
    return up


def diag_up(sp):
    up = np.zeros([n*m + n + 3, n*m + n + 3])
    up[sp][sp] = 1
    return up

def calc_r0(pert):
    sum1 = 0.0
    sum2 = 0.0
    check_zero = 0
    for i in range(n):
        if (pert[0][i] != 0):
            check_zero = 1
    if(check_zero == 0):
        return rmin
    for i in range(n):
        sum1 += pert[0][i] * w[0][i]
    max_for_ru = 0
    for j in range(1,m+1):
        sum2 = 0.0
        for i in range(n):
            sum2 += pert[0][i+j*n] * Dk[i][j]
        if(sum2 > max_for_ru):
            max_for_ru = sum2
    # print("sum1", sum1, type(sum1))
    # print("sum2", sum2, type(sum2))
    if(max_for_ru == 0):
        return rmin
        #print("max_for_ru is 0, pert is ", pert)
    ru = sum1 / max_for_ru
    rl = ((lt / (2 * le * p_elliniko)) ** (1/3)) #* (10**(-6))
    if ru < rmin:
        r0 = rmin
    elif (ru >= rmin) & (ru <= rmax):
        if(rl < rmin):
            r0 = minim_r0(pert, ru, rmin)
        elif (rl >= rmin)  & (rl <= ru):
            r0 = minim_r0(pert, ru, rl)
        else:
            r0 = ru
    else:
        if(rl < rmin):
            r0 = rmin
        elif (rl >= rmin)  & (rl <= rmax):
            r0 = rl
        else:
            r0 = rmax
    #r0 = ru
    return r0

def minim_r0(pert, r3, r4):
    if calculate_cost(pert, r3) < calculate_cost(pert, r4):
        return r3
    else:
        return r4
 
def initilization():
    C1 = np.zeros([q,q])
    a1 = np.zeros([q,1])
    a3 = np.zeros([1,q])
    a4 = np.zeros([2,1])
    a5 = np.zeros([1,2])
    #a6 = np.zeros([2,2])
    aq2 = np.zeros([q2,q2])
    #print("this is ckul" ,Ck_ul)
    g0 = []
    g0.append(w)
    g0.append(np.zeros([1,q - n]))
    g1 = np.block(
        g0
    )
    #print(g1)
    a0 = np.empty((q,1), float)
    for i in range(q):
        a0[i][0] = g1[0][i]
    #print("this is a0", a0)
    A0=[]
    row1 =[]
    row2 = []
    row3 = []
    fini = []
    row1.append(C1)
    row1.append(a0)
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
    #print("this is A0" , A0) #looks good
    # a2 einai o a2, g3 einai o adj tou a2
    g2 = []
    g2.append(np.zeros([1,q2 - 2]))
    g2.append(-10**(-6))
    g2.append(0)
    g3 = np.block(
        g2
    )

    a2 = np.empty((q2,1), float)
    for i in range(q2):
        a2[i][0] = g3[0][i]
    #print(a2)
    #print("this is a2", a2)
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
    #print("this is b2", b2 ) looks good

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
    #print("this is A2", A2) looks good


    a31 = []
    for i in range(m + 1):
        a31.append(np.identity(n))
    a31.append(np.zeros([n,3]))
    A3 = np.block(
        a31
    )
    #print( A3) looks good

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
    #print(A4) looks good

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
    #print(len(b4)) looks good



    b51 = []
    k5 = []
    b51.append(np.zeros([1,q3]))
    b51.append(-1)#*(10**(3)))
    b51.append(np.zeros([1,2]))
    k5 = np.block(
        b51
    )
    b5 = np.empty((q3+3,1), float)
    for i in range(q3+3):
        b5[i][0] = k5[0][i]
    #print(b5) looks good
    
    #edw 3ekinaei to ktisimo tou b0',o opios  einai o pt3
    pt1 = []
    pt2 = []

    for i in range(1, m+1):
        for j in range(n):
            pt1.append(ptx * dul[j][i])# *(10**(-3)))
            pt2.append(prx * ddl[j][i]) #*(10**(-3)))

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
    #print("edw einai o bo'", pt3)

    #gia ton b0
    b0 = np.empty((q+2,1), float)
    for i in range(n):
        b0[i][0] = 0
    for i in range(n*m):
        b0[i+n][0] = le * pt3[i][0] 
    b0[n+n*m] = 0
    b0[n+n*m + 1] = 0
    b0[n+n*m + 2] = lt #* (10**(-3))

    #print("edw einai o b0", b0)

    #gia ton A1
    A111 = []
    A11 = []
    A1 = []
    for i in range(m):
        A111.append(np.zeros([1,n]))
        for j in range(i):
            A111.append(np.zeros([1,n]))
        A111.append(Dk[ :, i])
        for coun in range(m-i-1):
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
    #print(A1)

    

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
    first_row.append(0.5*b0)
    #print("firstr row", first_row)
    final_array.append(first_row)
    second_row.append(0.5*b0t)
    second_row.append(0)
    #print("second row", second_row)
    final_array.append(second_row)
    B00 = np.block(
        final_array
    )
    # testing:
    # print("edw eibai o B0", B00)
    # print("B0 q row", B00[q])
    # print("B0 last row", B00[q+2])
    # print("B0 q column", B00[:,q])
    # print("B0 last column", B00[:,q+2])
    # print("length B0", len(B00))
    # print("size of B0", B00.size)
    # print("shape of B0", B00.shape)
    # counter1 = 0
    # for i in range(q4+1):
    #     for j in range(q4+1):
    #         if(B00[i][j] != 0):
    #             counter1 = counter1 + 1
    # print("counter is non zero elements in B0", counter1)

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
    #print("edw einai o B2", B20)
    # print("edw eibai o B2", B20)
    # print("B2 q row", B20[m*n+n+2])
    # print("B0 last row", B20[q+2])
    # print("B0 q column", B20[:,m*n+n+2])
    # print("B0 last column", B20[:,q+2])
    # print("length B2", len(B20))
    # print("size of B2", B20.size)
    # print("shape of B2", B20.shape)
    # counter1 = 0
    # for i in range(q4+1):
    #     for j in range(q4+1):
    #         if(B20[i][j] != 0):
    #             counter1 = counter1 + 1
    # print("counter is non zero elements in B2", counter1)

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
    #print("edw einai o B4", B40)
    # print("edw eibai o B4", B40)
    # print("B4 q row", B40[q3,q3])
    # print("B0 last row", B40[q+2])
    # print("B0 q column", B40[:,m*n+n+2])
    # print("B0 last column", B40[:,q+2])
    # print("length B2", len(B40))
    # print("size of B4", B40.size)
    # print("shape of B4", B40.shape)
    # counter1 = 0
    # for i in range(q4+1):
    #     for j in range(q4+1):
    #         if(B40[i][j] != 0):
    #             counter1 = counter1 + 1
    # print("counter is non zero elements in B4", counter1)

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
    # print("B5 last row", B50[q4])
    # print("B5 last column", B50[:,q4])
    #print("o B5", B50)
    # print("length B5", len(B50))
    # print("size of B5", B50.size)
    # print("shape of B5", B50.shape)
    # counter = 0
    # for i in range(q4+1):
    #     for j in range(q4+1):
    #         if(B50[i][j] != 0):
    #             counter = counter + 1
    # print("counter is non zero elements in B5", counter)
    #Gp 
    Gp_ol = []
    for j in range(p):
        Gp = []
        first_row = []
        second_row = []
        final_array = []

        up_adj = np.zeros([1, n*m + n + 3])
        up_adj[0][j] = 1
        first_row.append(diag_up(j))
        first_row.append((-0.5)*create_up(j))
        final_array.append(first_row)
        second_row.append((-0.5)*up_adj)
        second_row.append(0)
        final_array.append(second_row)
        Gp = np.block(
            final_array
        )
        Gp_ol.append(Gp)
    # print("this is gp 0", Gp_ol[0], Gp_ol[0][n*m+n+3], Gp_ol[0][:,n*m+n+3])
    # print("this is gp 1", Gp_ol[1], Gp_ol[1][n*m+n+3], Gp_ol[1][:,n*m+n+3])
    # print("this is gp last", Gp_ol[n*m+n-1], Gp_ol[n*m+n-1][n*m+n+3], Gp_ol[n*m+n-1][:,n*m+n+3])
    # print("checking", Gp_ol[0][n*m+n+3][0] == -0.5 )
    # print("this is gp 0", Gp_ol[0], Gp_ol[0][0], Gp_ol[0][n*m + n + 3])
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
    # print("edw einai o Hh 1")
    # print( Hh_ol[0]) #, Hh_ol[0])
    # print("Hh 0 q4" ,Hh_ol[0][q4]) #, Hh_ol[0])
    # print("Hh 0 q4" ,Hh_ol[0][:,q4]) #, Hh_ol[0])
    # print("Hh 1 q4" ,Hh_ol[1][q4]) #, Hh_ol[0])
    # print("Hh 1 q4" ,Hh_ol[1][:,q4]) #, Hh_ol[0])
    # print("edw einai o Hh 2")#, Hh_ol[1])
    # print(Hh_ol[1])
    # print("A1")
    # print(A1)

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
    # print("this is jj 0", Jj_ol[0][:,q4]) #, Jj_ol[1], Jj_ol[2])
    # print("this is jj 1", Jj_ol[1][:,q4])
    # print("this is jj 2", Jj_ol[2][:,q4])
    # print("this is jj n-1", Jj_ol[n-1][q4]) #[ :, i]
    # print("this is jj n-1", Jj_ol[n-1][ :, q4]) #[ :, i]
   # print("this is jj", Jj_ol[0], Jj_ol[0][0], Jj_ol[0][n*m + n + 3])
    return B00,B20,B40,B50,Gp_ol,Hh_ol,Jj_ol




def sdr_offloading(B00,B20,B40,B50,Gp_ol,Hh_ol,Jj_ol):     
    X = cp.Variable((q4+1,q4+1), symmetric=True)
    constraints= []
    constraints += [X >> 0]              # The operator >> denotes matrix inequality.
    constraints += [cp.trace(B40 @ X) == 0]
    constraints += [cp.trace(B20 @ X) <= 0]
    #rmin = 0.000001
    constraints += [cp.trace(B50 @ X) >= rmin]
    constraints += [cp.trace(B50 @ X) <= rmax]
    #constraints += [cp.trace(B50 @ X) == rmin]  #infeasable problem here
    constraints += [cp.trace(Jj_ol[i] @ X) == 1 for i in range(n)] #inaccurate, optimal otan einai comment
    constraints += [cp.trace(Hh_ol[i] @ X) <= 0 for i in range(m)]
    constraints += [cp.trace(Gp_ol[i] @ X) == 0 for i in range(p)]  #inacurate edw
    # #constraints += [X<= 1, X>= 0]   # Convex Relaxation 0<=x_i,y_{ij}<=1  #infeasable edw
    constraints += [ X>= 0]    
    # constraints += [ X[q4][q4] == 1] 

    prob = cp.Problem(cp.Minimize(cp.trace(B00 @ X)),
                    constraints)
    # prob.solve(solver="MOSEK", verbose=True)
    # prob.solve(solver="SCS")
    # prob.solve(solver="MOSEK")
    # prob.solve(solver="GUROBI",verbose=True)
    prob.solve(solver="SCS", verbose=True)
    # Print result.
    print("The SDR optimal value is", prob.value)
    #print("A solution X is")
    # np.set_printoptions(precision=3)

    #print (X.value.any() > 0 and X.value.any() < 1)
    rank = np.linalg.matrix_rank(X.value)
    print ("Rank of SDR solution is: ", rank)
    # return X.value

    ############# mapping to feasible solution 
    iteration_best = -1
    Xstar = X.value
    #print("Xstar is ", Xstar)
    Xstar = Xstar[:-4,:-4]
    #print(len(Xstar))
    minimum_obj = 10000000000000 
    solution=[]
    for l in range (iterations):
        ksi = np.random.multivariate_normal(np.zeros(p), Xstar,   tol=1e-6) #isws add size =100 arguement
        # print (ksi)
        xcandidate = 1/ (1 + np.exp(-m_elliniko*ksi))   # mapping function
        column_to_be_added = np.zeros([n,m+1])
        for i in range (n):
            for j in range (m):
                column_to_be_added[i,0] = xcandidate[i]
                column_to_be_added[i][j+1] = f'{xcandidate[n+m*i+j]:.20f}'
        b = np.zeros_like(column_to_be_added)
        b[np.arange(len(column_to_be_added)), column_to_be_added.argmax(1)] = 1
        pert = np.zeros([n*m+n+3, 1])
        for i in range(n):
            for j in range(m):
                pert[i][0] = b[i][0]
                pert[n+m*i+j][0] = b[i][j+1]
        pert = np.append(pert,1)
        pert = np.array([pert])
        #Y = np.transpose(pert)*pert
        # L = np.trace(B20 @ Y)
        #A = np.trace(B40 @ Y)
        # if L < 0 and A == 0 and all([np.trace(Jj_ol[i] @ Y) == 1 for i in range(n)]) and all ([np.trace(Hh_ol[i] @ Y) <= 0 for i in range(m)]) and all ([np.trace(Gp_ol[i] @ Y) == 0 for i in range(p)]):
            #candidate = np.trace(B00 @ Y)
        # print("pert here", pert[0])
        # print("end pert")
        candidate = calculate_cost(pert, calc_r0(pert)) #itan calc_r0(pert)
        if candidate < minimum_obj:
            minimum_obj= candidate
            solution = pert
            optimal_freq = calc_r0(pert)
            iteration_best = l
            # for iter in range(n):
            #     if pert[iter][0] != 0:
            #         ro = calc_r0(pert)
            #     else:
            #         ro = (rmin + rmax) / 2
            #Lbest = L/n
            #Amax= -A/n
    # else: 
        #     candidate = 1/n*np.trace(B0 @ Y)
        #     if candidate< minimum_obj:
        #         print ("ring the bell")
        #         print ("solution ", candidate)
        #         print ("solution ", pert)

    print ("iteration number to find optimal SDR solution: ", iteration_best)
    print ("minimum objective ", minimum_obj)  
    print ("pert ", pert)
    print ("solution array", solution)   
    print ("optimal freq", optimal_freq)
    sdr_solution = solution[0][:-1]
    print ("solution ", sdr_solution)
    print ("solution length is ", len(sdr_solution))

    # compute best L and A
    # print ("Lbest: ", Lbest)
    # print  ("Amax: ", Amax)
    return sdr_solution, optimal_freq


def random_compression(): 
    slist = np.random.uniform(0.1,1,size=(n,))
    return slist


def calculate_cost(solution, r):
    pert = solution#.tolist()
    #minimum_obj = 10000000000000 
    #pert.append(1)
    #pert = np.array([pert])
    #Y = np.transpose(pert)*pert
    #L = np.trace(B1 @ Y)
    #A = np.trace(B2 @ Y)
    #candidate = 1/n*np.trace(B0 @ Y)
    #minimum_obj= candidate
    #solution = pert 
    ecomp = 0
    for i in range(n):
        #print(pert[0])
        sum1 = p_elliniko * ((r*(10**6)) **z_elliniko) * pert[0][i] * Dk[i][0]
        ecomp = ecomp + sum1
    etr = 0
    for j in range(1,m+1):
        sum2 = 0
        for i in range(n):
            sum2 = sum2 + (ptx  * pert[0][i+j*n] * dul[i][j]) + (prx  * pert[0][i+j*n] * ddl[i][j])
        etr = etr + sum2 
    maxtk = 0
    for j in range(m+1):
        sum3 = 0
        for i in range(n):
            sum3 = sum3 + pert[0][i+j*n]*Dk[i][j] 
        if(sum3 > maxtk):
            maxtk = sum3
    e_syn = ecomp + etr
    total_cost = lt * maxtk + le * e_syn  
    # print (total_cost)
    # print("this is maxtk",  maxtk)
    # print("this is etr",  etr)
    # print("this is ecomp",  ecomp)
    # print("this is pert", pert[0])
    return total_cost

def total_cost_is(solution, r):
    pert = solution#.tolist()
    #minimum_obj = 10000000000000 
    #pert.append(1)
    #pert = np.array([pert])
    #Y = np.transpose(pert)*pert
    #L = np.trace(B1 @ Y)
    #A = np.trace(B2 @ Y)
    #candidate = 1/n*np.trace(B0 @ Y)
    #minimum_obj= candidate
    #solution = pert 
    ecomp = 0
    for i in range(n):
        sum1 = p_elliniko* 10**(-18) * (r**z_elliniko) * pert[i] * Dk[i][0]
        ecomp = ecomp + sum1
    etr = 0
    for j in range(1,m+1):
        sum2 = 0
        for i in range(n):
            sum2 = sum2 + (ptx  * pert[i+j*n] * dul[i][j]) + (prx  * pert[i+j*n] * ddl[i][j])
        etr = etr + sum2 
    maxtk = 0
    for j in range(m+1):
        sum3 = 0
        for i in range(n):
            sum3 = sum3 + pert[i+j*n]*Dk[i][j] 
        if(sum3 > maxtk):
            maxtk = sum3
    e_syn = ecomp + etr
    total_cost = lt * maxtk + le * e_syn  
    # print (total_cost)
    # print("this is maxtk",  maxtk)
    # print("this is etr",  etr)
    # print("this is ecomp",  ecomp)
    # print("this is pert", pert[0])
    return total_cost

def main():
    sdr_solution = [0]
    # slist = random_compression()
    # Lcurrent = 1000000
    # Acurrent = -10000000
    # epsilon = 0.01
    # counter = 0 
    # prev_sol = sdr_solution
    B0,B1,B2,B3,B4,B5,B6 = initilization()
    sdr_solution, r_opt = sdr_offloading(B0,B1,B2,B3,B4,B5,B6) 
    # while (True):
    #     # print ("\n new iteration number: ", counter)
    #     prev_sol = sdr_solution
    #     B0,B1,B2,B3,B4,B5,B6 = initilization(slist)
    #     sdr_solution, Lbest, Amax = sdr_offloading(B0,B1,B2,B3,B4,B5,B6) 
    #     # print (sdr_solution)
    #     # sdr_solution = random_offloading(B0,B1,B2,B3,B4)
    #     # sdr_solution, Lbest, Amax  = implement_brute_force(B0,B1,B2,B3,B4,B5)
    #     # slist = random_compression()
    #     # print ("Total Cost of Solution: ", calculate_cost(sdr_solution,B0,B1,B2,B3,B4))
    #     slist = scaling_problem(sdr_solution)
    #     # if not((sdr_solution == prev_sol).all()) and counter != 0 :
    #     #     print("different solution found:")
    #     #     print ("new solution:" , sdr_solution)
    #     # #     print ("previous solution:" ,prev_sol)
    #     # if (Lbest-alpha*Amax <= Lcurrent-alpha*Acurrent + epsilon) and (Lbest-alpha*Amax >= Lcurrent-alpha*Acurrent - epsilon):
    #     #     # print ("condition2")
    #     #     break; 
    #     # Lcurrent = Lbest
    #     # Acurrent = Amax
    #     counter +=1
    #     break;
    print("sdr solution is,", sdr_solution )
    r_final = r_opt * (10**6)
    print("final optimal freq is ", r_final)
    costing = total_cost_is(sdr_solution,r_final)
    print ("cost is", costing)
    return 

if __name__ == "__main__":
    main()
