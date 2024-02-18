#!/usr/bin/env python3
import cvxpy as cp
import numpy as np
import itertools
import random
from math import exp, log

###################### basic settings
m = 3
n = 4  #kanonika 10
p = n*m + n 
q = m*n + n + 1
q2 = m*n + n + 2
q3 = m*n + n
q4 = m*n + n + 3
iterations = 150
np.random.seed(5)

###################### constants
d = np.random.uniform(1,2,size=(n,))
s_initial = np.random.uniform(0.1,1,size=(n,))
# b = np.random.uniform(15,95,size=(m,))
R = np.random.uniform(5,10,size=(n,m))
bmax = np.random.uniform(n/4,n/3,size=(m,))
# print (bmax)
Lmax = 0.9 # 0.3 #seconds
Amin = 0.7 #0.7 
zlocal = 0.011
ksilocal = 0.7676
zedge = 0.02862
ksiedge = 0.06706
alpha = 1 # change from 0.1 to 500 
ptx = 1.285
prx = 1.181
le = 0.5
lt = 1 - le
L = 100
s_elliniko = 150 
rmin = (400-s_elliniko) * (10**6) 
rmax = (400+s_elliniko) * (10**6)
m_elliniko = 10
p_elliniko = 1.25 * (10**(-16))  #-26 kanonika
z_elliniko = 3
Ck_ul = np.random.uniform(10,20,size=(m,))
Ck_dl = np.random.uniform(10,20,size=(m,))
ai = np.empty((n), float) #make 0.1 0.2 ...
for i in range(n):
    ai[i] = (0.1 + i*0.1) * (10**6)
#print("this is ai", ai)
w = np.empty((1,n), float)
for i in range(n):
    w[0][i] = 330 * ai[i]
#print("this is w", w)
bi = np.empty((n), float)
for i in range(n):
    bi[i] = 0.2 * ai[i]
r_k = np.empty((m), float)
for i in range(m):
    r_k[i] = 2 * (10**9)
# r_k[0] = 1
# r_k[1] = 2
# r_k[2] = 2.2
dul = np.empty((n,m), float)
ddl = np.empty((n,m), float)
Dk = np.empty((n,m), float)

for i in range(n):
    for j in range(m):
        dul[i][j] = ai[i] / Ck_ul[j]
        ddl[i][j] = bi[i] / Ck_dl[j]
        Dk[i][j] = dul[i][j] + ddl[i][j] + (w[0][i] / r_k[j])


#################### functions
def gompertz_local (s, remote): # remote == True for edge gompertz function. 
    if remote == True:
        a = 0.95
        b = 5
        c = 5
    else: 
        a = 0.65  #TODO fix this with something that can be tuned
        b = 5 
        c = 5
    g = a*exp(-b*exp(-c*s))
    return g

def calculate_hyperplane_approximation(trans_time): #TODO substitute with gompertz variables
    x_vals = np.linspace(0.0001, 0.94, 50) # generate 50 evenly spaced values between 0 and 0.95
    y_vals = np.linspace(0.0001, 10, 50) # generate 50 evenly spaced values between 0 and 1

    x, y = np.meshgrid(x_vals, y_vals) # create a grid of (x,y) points

    z = trans_time*0.5*np.log(5/np.log(0.95/x)) - y # compute the function values at each (x,y) point

    A = np.column_stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())]) # create the design matrix
    coeffs, _, _, _ = np.linalg.lstsq(A, z.ravel(), rcond=None) # compute the coefficients using least squares regression

    a, b, c = coeffs 

    return a,b,c

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
    for i in range(n):
        sum2 += pert[0][i] * Dk[i][0]
    # print("sum1", sum1, type(sum1))
    # print("sum2", sum2, type(sum2))
    ru = sum1 / sum2
    rl = (lt / (2 * le * p_elliniko)) ** (1/3)
    if ru < rmin:
        r0 = rmin
    elif (ru >= rmin) & (ru <= rmax):
        if(rl < rmin):
            r0 = minim_r0(pert, ru, rmin)
        elif (rl >= rmin)  & (rl <= rmax):
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
 
def initilization(s):
    C1 = np.zeros([q,q])
    a1 = np.zeros([q,1])
    a3 = np.zeros([1,q])
    a4 = np.zeros([2,1])
    a5 = np.zeros([1,2])
    #a6 = np.zeros([2,2])
    aq2 = np.zeros([q2,q2])

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
    #print("this is A0" , len(A0)) looks good
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
    b51.append(-1)
    b51.append(np.zeros([1,2]))
    k5 = np.block(
        b51
    )
    b5 = np.empty((q3+3,1), float)
    for i in range(q3+3):
        b5[i][0] = k5[0][i]
    #print(b5) looks good
    #EDW EIXA MEINEI STO TESTING
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
    #print("edw einai o bo'", pt3)

    #gia ton b0
    b0 = np.empty((q+2,1), float)
    for i in range(n):
        b0[i][0] = 0
    for i in range(n*m):
        b0[i+n][0] = le * pt3[i][0]
    b0[n+n*m] = 0
    b0[n+n*m + 1] = 0
    b0[n+n*m + 2] = lt 

    #print("edw einai o b0", b0)

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
    final_array.append(first_row)
    second_row.append(0.5*b0t)
    second_row.append(0)
    final_array.append(second_row)
    B00 = np.block(
        final_array
    )
    #print("edw eibai o B0", B00)


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
    #print("edw einai o B5", B50)

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
        first_row.append(-0.5*create_up(j))
        final_array.append(first_row)
        second_row.append(-0.5*up_adj)
        second_row.append(0)
        final_array.append(second_row)
        Gp = np.block(
            final_array
        )
        Gp_ol.append(Gp)

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

    return B00,B20,B40,B50,Gp_ol,Hh_ol,Jj_ol

def scaling_problem(sdr_solution):
    # print("Start of scaling Problem")
    x_variables = sdr_solution[0:n]
    #calculate local average for Latency and accuracy
    local_executed_requests_count = x_variables.tolist().count(1)
    if local_executed_requests_count!=0:
        average_local_latency = sum((ksilocal*x*y+zlocal for x,y in zip(d,x_variables.tolist())))/local_executed_requests_count
        average_local_accuracy = gompertz_local(1,False)
    else: 
        average_local_latency = 0 
        average_local_accuracy = 0 


    sdr_solution = sdr_solution[n:] # remove first n elements for x_i variables and the last one
    
    y = [sdr_solution[j:j + m] for j in range(0, len(sdr_solution), m)]  # transform 1-d array to 2-d arrays of m columns
    y = [l.tolist() for l in y]   # list of arrays to list of lists
    
    # cpvxpy variables
    u = cp.Variable(m) # regards the variable t of time 
    w = cp.Variable(n) # regards the substitute of gompertz function 
    constraints= []
    objective = 0 
    # this is to add all log_sum_exp 
    for i in range (n): 
        for j in range (m): 
            objective += y[i][j]*u[j] - (alpha*y[i][j])*w[i] 
    # this is to add the constraints

    temp = 0 
    temp1 = 0 
    for j in range (m): 
        for i in range (n): 
            temp += y[i][j] * u[j]    
            temp1 += y[i][j] * w[i] 
            constraints += [w[i] >= 0.95*exp(-5)]    #TODO substitute with gompertz variables
            constraints += [w[i] <= 0.95*exp(-5*exp(-5))]  #TODO substitute with gompertz variables
            constraints += [cp.exp(u[j]) >= 0.1]

            a,b,c = calculate_hyperplane_approximation(d[i] / R[i][j])
            constraints += [a*w[i]+b*u[j]+c<=0]

            # first_term = (d[i] / R[i][j])* log((exp(1)+1)/2) - exp(-1.15) - (-1.15)*(-exp(-1.15)) - (exp(1)+1)/2*((d[i] / R[i][j])/(exp(1)+1))
            # constraints += [first_term-cp.exp(-1.15)*u[j] + (d[i] / R[i][j])/(exp(1)+1)*w[i]<=0] 
            # constraints += [y[i][j] * (d[i] / R[i][j]) * cp.log(w[i]) <= cp.exp(u[j])]#  y[i][j] * (c * d[i] / R[i][j]) * cp.log((-cp.log(a)+w)/b)
    constraints += [temp <= Lmax*n-average_local_latency*local_executed_requests_count]
    constraints += [temp1 >= Amin*n-average_local_accuracy*local_executed_requests_count]
    # objective = cp.log_sum_exp(objective)
    prob = cp.Problem(cp.Minimize(objective),constraints)
    # prob.solve(solver=cp.SCS, qcp=True ,low=1, high=5)#, verbose= True)
    prob.solve(solver="MOSEK",  qcp=True)#, verbose=True)

    # prob.solve(solver="SCS",verbose=True)
    # Print result.
    # print("The Convex Program for s optimal value is", prob.value)
    # print("A solution for u is")
    # print(u.value)
    # print("A solution for w is")
    # print(w.value)
    # tlist = u.value

    wlist = w.value
    slist = []
    for item in wlist:
        slist.append(1/5*log(5/(log(0.95/item)))) #TODO include variables of gompertz
    # print("A solution for s is", slist)
    return slist


def sdr_offloading(B00,B20,B40,B50,Gp_ol,Hh_ol,Jj_ol):     
    X = cp.Variable((q4+1,q4+1), symmetric=True)
    constraints= []
    constraints += [X >> 0]              # The operator >> denotes matrix inequality.
    constraints += [cp.trace(B40 @ X) == 0]
    constraints += [cp.trace(B20 @ X) <= 0]
    constraints += [cp.trace(B50 @ X) <= rmax]
    constraints += [cp.trace(B50 @ X) >= rmin]
    constraints += [cp.trace(Jj_ol[i] @ X) == 1 for i in range(n)]
    constraints += [cp.trace(Hh_ol[i] @ X) <= 0 for i in range(m)]
    constraints += [cp.trace(Gp_ol[i] @ X) == 0 for i in range(p)]
    # constraints += [X<= 1, X>= 0]   # Convex Relaxation 0<=x_i,y_{ij}<=1
    # constraints += [ X>= 0]    

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
    # rank = np.linalg.matrix_rank(X.value)
    # print ("Rank of SDR solution is: ", rank)
    # return X.value

    ############# mapping to feasible solution 
    iteration_best = -1
    Xstar = X.value
    print("Xstar is ", Xstar)
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
        Y = np.transpose(pert)*pert
        # L = np.trace(B20 @ Y)
        #A = np.trace(B40 @ Y)
        # if L < 0 and A == 0 and all([np.trace(Jj_ol[i] @ Y) == 1 for i in range(n)]) and all ([np.trace(Hh_ol[i] @ Y) <= 0 for i in range(m)]) and all ([np.trace(Gp_ol[i] @ Y) == 0 for i in range(p)]):
            #candidate = np.trace(B00 @ Y)
        candidate = calc_r0(pert)
        if candidate < minimum_obj:
            minimum_obj= candidate
            solution = pert
            optimal_freq = candidate
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

    # compute best L and A
    # print ("Lbest: ", Lbest)
    # print  ("Amax: ", Amax)
    return sdr_solution, optimal_freq

def random_offloading(B0,B1,B2,B3,B4):
    condition = True
    while (condition): 
        condition = False
        xlist =  [random.randint(0, 1) for _ in range(n)]
        ylist = [] 
        for i, val in enumerate(xlist):
            if val == 0:
                sublist = [0] * m
                idx = np.random.randint(0, m)  # choose random index to set to 1
                sublist[idx] = 1
            else:
                sublist = [0] * m
            ylist.append(sublist)
        solution = np.concatenate((xlist, np.array(ylist).flatten()))
        for j in range(m):
            check_bmax = 0 
            for sublist in ylist:
                check_bmax += sublist[j]
            if check_bmax >= bmax[j]:
                condition= True
        total_cost, L, A = calculate_cost(solution,B0,B1,B2,B3,B4)
        if L >Lmax or A <= Amin:
            condition= True
        # print (solution)
    # print (check_bmax)
    # print (bmax)
    return solution

def random_compression(): 
    slist = np.random.uniform(0.1,1,size=(n,))
    return slist

def offload_local():
    xlist =  [1]*n
    ylist = [] 
    for i, val in enumerate(xlist):
        sublist = [0] * m
        ylist.append(sublist)
    solution = np.concatenate((xlist, np.array(ylist).flatten()))
    return solution

def offload_remote(): 
    xlist = [0]*n
    ylist = [] 
    for i, val in enumerate(xlist):
        sublist = [0] * m
        idx = np.random.randint(0, m)  # choose random index to set to 1
        sublist[idx] = 1
        ylist.append(sublist)
    solution = np.concatenate((xlist, np.array(ylist).flatten()))
    return solution

def implement_brute_force(B0,B1,B2,B3,B4,B5): 
    ############## Implement brute force solution....
    A = lst = [list(i) for i in itertools.product([0, 1], repeat=n+n*m)]
    minimum_obj = 10000000000000 
    solution=[]
    # print (len(A))
    for pert in A:
        pert.append(1)
        pert = np.array([pert])
        Y = np.transpose(pert)*pert
        L = np.trace(B1 @ Y)
        A = np.trace(B2 @ Y)
        if L < Lmax*n and A <= -Amin * n and all([np.trace(B3[i] @ Y) == 1 for i in range(n)]) and all ([np.trace(B4[i] @ Y) <= bmax[i] for i in range(m)]):
            candidate = 1/n*np.trace(B0 @ Y)
            if candidate< minimum_obj:
                minimum_obj= candidate
                solution = pert 
                # print ("L: ", L/n)
                # print  ("A: ", -A/n)
    # print ("solution ", solution)
    # print ("minimum objective ", minimum_obj)
    return solution[0], L, A

#geniki synartisi kostous gia X kai tin trexw kai gia to r0 kai gia to X_0
def calculate_cost(solution, r):
    pert = solution.tolist()
    #minimum_obj = 10000000000000 
    pert.append(1)
    pert = np.array([pert])
    #Y = np.transpose(pert)*pert
    #L = np.trace(B1 @ Y)
    #A = np.trace(B2 @ Y)
    #candidate = 1/n*np.trace(B0 @ Y)
    #minimum_obj= candidate
    solution = pert 
    ecomp = 0
    for i in range(n):
        sum1 = p_elliniko * (r**z_elliniko) * pert[0][i] * Dk[i][0]
        ecomp = ecomp + sum1
    etr = 0
    for j in range(1,m):
        sum2 = 0
        for i in range(n):
            sum2 = sum2 + (ptx  * pert[0][j] * dul[i][j]) + (prx  * pert[0][j] * ddl[i][j])
        etr = etr + sum2 
    maxtk = 0
    for j in range(m):
        sum3 = 0
        for i in range(n):
            sum3 = sum3 + pert[0][j]*Dk[i][j] 
        if(sum3 > maxtk):
            maxtk = sum3
    e_syn = ecomp + etr
    total_cost = lt * maxtk + le * e_syn  
    # print (total_cost)
    return total_cost

def main():
    sdr_solution = [0]
    slist = random_compression()
    Lcurrent = 1000000
    Acurrent = -10000000
    epsilon = 0.01
    counter = 0 
    prev_sol = sdr_solution
    B0,B1,B2,B3,B4,B5,B6 = initilization(slist)
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
    total_cost = calculate_cost(sdr_solution,r_opt)
    print (total_cost)
    return 

if __name__ == "__main__":
    main()
