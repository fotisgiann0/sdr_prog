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
n = 12
p = n*m + n 
q = m*n + n + 1
q2 = m*n + n + 2
q3 = m*n + n
q4 = m*n + n + 3
iterations = 100
np.random.seed(5)

###################### constants
d = np.random.uniform(1,2,size=(n,))
s_initial = np.random.uniform(0.1,1,size=(n,))
R = np.random.uniform(5,10,size=(n,m))
bmax = np.random.uniform(n/4,n/3,size=(m,))
 
ptx = 1.285
prx = 1.181
le = 0.5
lt = 1 - le
L = 100
s_elliniko = 150 
rmin = 400 
rmax = 800 
m_elliniko = 10
p_elliniko = 1.25 * (10**(-8)) 
z_elliniko = 3
pcomp = 0.8 #p_elliniko * (rmin**3)
Ck_ul = [25, 35, 30]#, 12]#np.random.uniform(10,20,size=(m+1,))  
Ck_dl = [40, 43, 27]#, 15]#np.random.uniform(10,20,size=(m+1,))  
ai = np.empty((n), float) 
for i in range(n):
    ai[i] =  0.5 #(0.1 + i*0.1) # 0.5

w = np.empty((1,n), float)
for i in range(n):
    w[0][i] = 330 * ai[i]

bi = np.empty((n), float)
for i in range(n):
    bi[i] = 0.2 * ai[i]
r_k = np.empty((m+1), float)
for i in range(m+1):
    r_k[i] = 0 #2 * (10**9)
r_k[0] = 400 #400 * (10**6)
r_k[1] = 2000#2 * (10**9)
r_k[2] = 2200#2.2 * (10**9)
#r_k[3] = 2000#2.2 * (10**9)
#r_k[4] = 2000#2.2 * (10**9)
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

        
def create_up(sp):
    up = np.zeros([n*m + n + 1, 1])
    up[sp][0] = 1
    #print(up)
    return up


def diag_up(sp):
    up = np.zeros([n*m + n + 1, n*m + n + 1])
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

    a31 = []
    for i in range(m + 1):
        a31.append(np.identity(n))
    a31.append(np.zeros([n,1]))
    A2 = np.block(
        a31
    )


    pt1 = []
    pt2 = []

    for i in range(1, m+1):
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
    #print("edw einai o bo'", pt3)

    #gia ton b0
    b0 = np.empty((q,1), float)
    for i in range(n):
        b0[i][0] = le *pcomp * Dk[i][0]
    for i in range(n*m):
        b0[i+n][0] = le * pt3[i][0] 
    b0[n+n*m] = lt 


#     #print("edw einai o b0", b0)

    #gia ton A1
    A111 = []
    A11 = []
    A1 = []
    A111.append(Dk[ :, 0])
    for i in range(m):
        A111.append(np.zeros([1,n]))
    A111.append(-1)
    A11.append(A111)
    A111 = []
    for i in range(m):
        A111.append(np.zeros([1,n]))
        for j in range(i):
            A111.append(np.zeros([1,n]))
        A111.append(Dk[ :, i])
        for coun in range(m-i-1):
            A111.append(np.zeros([1,n]))
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
    b0t = np.empty((1, q), float)
    for i in range(q):
        b0t[0][i] = b0[i][0]



    first_row.append(np.zeros([q,q]))
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
 
    #Gp 
    Gp_ol = []
    for j in range(p):
        Gp = []
        first_row = []
        second_row = []
        final_array = []

        up_adj = np.zeros([1, n*m + n + 1])
        up_adj[0][j] = 1
        first_row.append(diag_up(j))
        first_row.append((-1)*(0.5)*create_up(j))
        final_array.append(first_row)
        second_row.append((-1)*(0.5)*up_adj)
        second_row.append(0)
        final_array.append(second_row)
        Gp = np.block(
            final_array
        )
        Gp_ol.append(Gp)
    # for testing:
    # for k in range(p):     
    #     counter = 0
    #     for i in range(q+1):
    #         for j in range(q+1):
    #             if(Gp_ol[k][i][j] != 0):
    #                 counter = counter + 1
    #     print("counter is non zero elements in Gp", k, counter)
    # print("this is gp 0", Gp_ol[0], Gp_ol[0][0],  Gp_ol[0][p+1], Gp_ol[0][:,p+1])
    # print("gp is ", Gp_ol[0].shape)
    # print("this is gp 1", Gp_ol[1], Gp_ol[1][1],  Gp_ol[1][p+1], Gp_ol[1][:,p+1])
    # print("gp is ", Gp_ol[1].shape)
    # print("this is gp 1", Gp_ol[p-1], Gp_ol[p-1][p-1],  Gp_ol[p-1][p+1], Gp_ol[p-1][:,p+1])
    # print("gp is ", Gp_ol[p-1].shape)
    # print("this is gp 1", Gp_ol[1], Gp_ol[1][n*m+n+3], Gp_ol[1][:,n*m+n+3])
    # print("this is gp last", Gp_ol[n*m+n-1], Gp_ol[n*m+n-1][n*m+n+3], Gp_ol[n*m+n-1][:,n*m+n+3])
    # print("checking", Gp_ol[0][n*m+n+3][0] == -0.5 )
    # print("this is gp 0", Gp_ol[0], Gp_ol[0][0], Gp_ol[0][n*m + n + 3])
    Hh_ol = []
    for j in range(m+1):
        Hh = []
        first_row = []
        second_row = []
        final_array = []
        hh1 = np.zeros([n+ m*n + 1 ,1])  #grammi h tou A1 kai adj tis grammis
        hh2 = np.zeros([1,n+ m*n + 1])
        for i in range(n+ m*n + 1):
            hh1[i][0] = A1[j][i]
            hh2[0][i] = A1[j][i]


        first_row.append(np.zeros([q,q]))
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
     # for testing:
    # print("edw einai o Hh 2")
    # print( Hh_ol[2]) #, Hh_ol[0])
    # print("Hh 0 q" ,Hh_ol[0][q]) #, Hh_ol[0])
    # print("Hh 0 q" ,Hh_ol[0][:,q]) #, Hh_ol[0])
    # print("Hh 1 q4" ,Hh_ol[1][q]) #, Hh_ol[0])
    # print("Hh 1 q4" ,Hh_ol[1][:,q]) #, Hh_ol[0])
    # print("Hh 2 q4" ,Hh_ol[2][q]) #, Hh_ol[0])
    # print("Hh 2 q4" ,Hh_ol[2][:,q]) #, Hh_ol[0])
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
        jj1 = np.zeros([m*n + 1 + n,1])  #grammi jj_row tou A3 kai adj tis grammis
        jj2 = np.zeros([1,m*n + n + 1])
        for i in range(m*n + 1 + n):
            jj1[i][0] = A2[j][i]
            jj2[0][i] = A2[j][i]


        first_row.append(np.zeros([q,q]))
        first_row.append(0.5*jj1)
        final_array.append(first_row)
        second_row.append(0.5*jj2)
        second_row.append(0)
        final_array.append(second_row)
        Jj = np.block(
            final_array
        )
        Jj_ol.append(Jj)
     # for testing:
    # print("this is jj 0", Jj_ol[0]) #, Jj_ol[1], Jj_ol[2])
    # print("this is jj 0", Jj_ol[0][:,q])
    # print("this is jj 0", Jj_ol[0][q])
    # print("this is jj 1", Jj_ol[9])
    # print("this is jj 1", Jj_ol[9][q])
    # print("this is jj 1", Jj_ol[9][:,q])   
    # print("this is jj 2", Jj_ol[2][:,q4])
    # print("this is jj n-1", Jj_ol[n-1][q4]) #[ :, i]
    # print("this is jj n-1", Jj_ol[n-1][ :, q4]) #[ :, i]
   # print("this is jj", Jj_ol[0], Jj_ol[0][0], Jj_ol[0][n*m + n + 3])
    return B00,Gp_ol,Hh_ol,Jj_ol

def sdr_offloading(B00,Gp_ol,Hh_ol,Jj_ol):     
    X = cp.Variable((q+1,q+1), symmetric=True)
    constraints= []
    constraints += [X >> 0]              # The operator >> denotes matrix inequality.
    constraints += [cp.trace(Jj_ol[i] @ X) == 1 for i in range(n)] #inaccurate, optimal otan einai comment
    constraints += [cp.trace(Hh_ol[i] @ X) <= 0 for i in range(m+1)]
    constraints += [cp.trace(Gp_ol[i] @ X) == 0 for i in range(p)]  #inacurate edw
    # #constraints += [X<= 1, X>= 0]   # Convex Relaxation 0<=x_i,y_{ij}<=1  #infeasable edw
    constraints += [ X>= 0]    
    constraints += [ X[q][q] == 1] 

    prob = cp.Problem(cp.Minimize(cp.trace(B00 @ X)),
                    constraints)
    # prob.solve(solver="MOSEK", verbose=True)
    # prob.solve(solver="SCS")
    # prob.solve(solver="MOSEK")
    # prob.solve(solver="GUROBI",verbose=True)
    prob.solve(solver="SCS")#, verbose=True)
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
    Xstar = Xstar[:-2,:-2]
    # print(len(Xstar))
    # print("p", p)
    minimum_obj = 10000000000000 
    solution=[]
    for l in range (iterations):
        ksi = np.random.multivariate_normal(np.zeros(p), Xstar,   tol=1e-6) 
        # print (ksi)
        xcandidate = 1/ (1 + np.exp(-m_elliniko*ksi))   # mapping function
        column_to_be_added = np.zeros([n,m+1])
        for i in range (n):
            for j in range (m):
                column_to_be_added[i,0] = xcandidate[i]
                column_to_be_added[i][j+1] = f'{xcandidate[n+m*i+j]:.20f}'
        b = np.zeros_like(column_to_be_added)
        b[np.arange(len(column_to_be_added)), column_to_be_added.argmax(1)] = 1
        pert = np.zeros([n*m+n+1, 1])
        for i in range(n):
            for j in range(m):
                pert[i][0] = b[i][0]
                pert[n+m*i+j][0] = b[i][j+1]
        pert = np.append(pert,1)
        pert = np.array([pert])
        Y = np.transpose(pert)*pert
        # L = np.trace(B20 @ Y)
        #A = np.trace(B40 @ Y)
        list1 = []
        for i in range(n):
            sum1 = 0
            for j in range(m+1):
                sum1 = sum1 + pert[0][i+j*n]
            list1.append(sum1)
        if (list1[i]== 1 for i in range(n)):
        #candidate = np.trace(B00 @ Y)
        # print("pert here", pert[0])
        # print("end pert")
            candidate = calculate_cost(pert) #itan calc_r0(pert)
            if candidate < minimum_obj:
                minimum_obj= candidate
                solution = pert
                #optimal_freq = candidate
                iteration_best = l


    print ("iteration number to find optimal SDR solution: ", iteration_best)
    print ("minimum objective ", minimum_obj)  
    # print ("pert ", pert)
    # print ("solution array", solution)   
    #print ("optimal freq", optimal_freq)
    sdr_solution = solution[0][:-1]
    # print ("solution ", sdr_solution)
    # print ("solution length is ", len(sdr_solution))


    return sdr_solution

def random_compression(): 
    slist = np.random.uniform(0.1,1,size=(n,))
    return slist

def random_offloading():
    solution = np.zeros( [1, n*(m+1)])
    for i in range(n):
        rand_idx = random.randint(0, m)
       # print(rand_idx)
        for j in range(m+1):
            if(j==rand_idx):
                solution[0][j*n+i] = 1
            else:
                solution[0][j*n+i] = 0
    #testing it
    # list1 = []
    # for i in range(n):
    #     sum1 = 0
    #     for j in range(m+1):
    #         sum1 = sum1 + solution[0][i+j*n]
    #     list1.append(sum1)
    # if (list1[i]== 1 for i in range(n)):
    #     print("true")
    return solution[0]

def local_processing():
    solution = np.zeros( [1, n*(m+1)])
    for i in range(n):
        solution[0][i] = 1
    return solution[0]


def calculate_cost(solution):
    pert = solution#.tolist()

    ecomp = 0
    for i in range(n):
        ecomp = ecomp + pcomp * pert[0][i] * Dk[i][0]
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

    return total_cost

def total_cost_is(solution):
    pert = solution
    ecomp = 0
    sum1 = 0
    for i in range(n):
        sum1 = pcomp * pert[i] * Dk[i][0]
        ecomp = ecomp + sum1
    etr = 0
    for j in range(1,m+1):
        sum2 = 0
        #print("j is ", j)
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
    e_syn = (ecomp + etr) * 10
    total_cost = lt * maxtk * 10 + le * e_syn  
    # print (total_cost)
    # print("this is etr",  etr)
    # print("this is ecomp",  ecomp)
    # print("this is execution latency",  maxtk* 10)
    # print("this is energy consumption",  e_syn)

    return total_cost

def offloading_portion(solution):
    counter = 0
    for i in range(n):
        if solution[i] == 1:
            counter = counter + 1
    portion = (n - counter) / n
    return portion

def main():
    sdr_solution = [0]
    local_list = local_processing()
    B0,B1,B2,B3 = initilization()
    sdr_solution = sdr_offloading(B0,B1,B2,B3) 

    sum_rand = 0
    for i in range(100):
        rand_list = random_offloading()
        sum_rand += total_cost_is(rand_list)
    rand_cost = sum_rand / 100
    print("sdr solution is:", sdr_solution )
    #print("sdr solution length:", len(sdr_solution))
    #r_final = r_opt * (10**6)
    print("pcomp is", pcomp)
    print("offloading portion:", offloading_portion(sdr_solution))
    costing = total_cost_is(sdr_solution)
    print ("cost is", costing)
    print("random assign cost:", rand_cost)
    print("local processing cost:", total_cost_is(local_list))
    return 

if __name__ == "__main__":
    main()