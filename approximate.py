#!/usr/bin/env python3
import cvxpy as cp
import numpy as np
import itertools
import random
from math import exp, log

###################### basic settings
n = 4
m = 2
p = n*m+n
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

def initilization(s):
    ################ auxiliary variables
    li = np.empty((n,1), float)
    Li = np.empty((n,1), float)
    gi = np.empty((n,1), float)
    dij = np.empty((n*m,1), float)
    Dij = np.empty((n*m,1), float)
    Gij = np.empty((n*m,1), float)
    index = 0
    for i in range (n):
        li[i][0] = ksilocal*d[i]+zlocal
        Li[i][0] = ksilocal*d[i]+zlocal-alpha * gompertz_local(1,False)
        gi[i][0] = gompertz_local(1,False)
        for j in range (m):
            dij[index][0] = s[i]*d[i] / R[i][j]
            Dij[index][0] = s[i]*d[i] / R[i][j] - alpha * gompertz_local(s[i],True)
            Gij[index][0] = gompertz_local(s[i],True)
            index +=1 
    b0 = np.concatenate((Li,Dij),axis=0)
    c0 = np.concatenate((li,dij),axis=0)
    d0 = np.concatenate((gi,Gij),axis=0)
    ########### Matrix A_0
    B=[]
    A1 = np.zeros([n,n])
    A2 = np.zeros([n,m])
    A3 = np.zeros([m,n])
    A4 = np.diag(np.full(m,ksiedge))

    # print (Dij)

    first_row = []
    final_matrix = []
    for j in range(n+1):
        if j == 0:
            first_row.append(A1)
        else:
            first_row.append(A2) 
    final_matrix.append(first_row)
    row = []
    for j in range(n+1):
        if j == 0:
            row.append(A3)
        else:
            row.append(A4) 
    for j in range(n):
        final_matrix.append(row)
    B = np.block(
        final_matrix
    )
    # print (B)
    # print (B.shape)

    ########### Matrix B_0 for objective function
    column_to_be_added = np.full((p,1),0.5*b0)
    B0 = np.hstack((B,column_to_be_added))
    column_to_be_added = np.append(column_to_be_added, [0])
    B0 = np.vstack((B0,column_to_be_added))


    ########### Matrix B_1 for constraint 9b    
    column_to_be_added = np.full((p,1),0.5*c0)
    B1 = np.hstack((B,column_to_be_added))
    column_to_be_added = np.append(column_to_be_added, [0])
    B1 = np.vstack((B1,column_to_be_added))

    ####################### matrix B2 for constraint 9c 
    B2 = np.zeros([p,p])
    column_to_be_added = np.full((p,1),-0.5*d0)
    B2 = np.hstack((B2,column_to_be_added))
    column_to_be_added = np.append(column_to_be_added, [0])
    B2 = np.vstack((B2,column_to_be_added))
    # print (B2)


    ####################### matrix B3 for constraint 9d 
    B3=[]
    for i in range (n): # we have n constraints of this form
        A3 = np.zeros([p,p])
        column_to_be_added = np.zeros([p,1])
        column_to_be_added[i,0] = 0.5
        for k in range (m):
            column_to_be_added[i*m+n+k,0] = 0.5 
        A3 = np.hstack((A3,column_to_be_added))
        column_to_be_added = np.append(column_to_be_added, [0])
        A3 = np.vstack((A3,column_to_be_added))
        B3.append(A3)

    ####################### matrix B4 for constraint 9E 
    B4=[]
    for i in range (m):   # we have m constraints of this form
        A4 = np.zeros([p,p])
        column_to_be_added = np.zeros([p,1])
        for k in range (n):
            column_to_be_added[k*m+n+i,0] = 0.5 
        A4 = np.hstack((A4,column_to_be_added))
        column_to_be_added = np.append(column_to_be_added, [0])
        A4 = np.vstack((A4,column_to_be_added))
        B4.append(A4)

    ####################### matrix B5 for constraint 9f
    B5=[]
    for i in range (p):   # we have p constraints of this form
        A5 = np.zeros((p,p))
        A5[i][i] = 1
        column_to_be_added = np.zeros([p,1])
        column_to_be_added[i] = -0.5 
        A5 = np.hstack((A5,column_to_be_added))
        column_to_be_added = np.append(column_to_be_added, [0])
        A5 = np.vstack((A5,column_to_be_added))
        B5.append(A5)

    return B0,B1,B2,B3,B4,B5

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


def sdr_offloading(B0,B1,B2,B3,B4,B5):     
    X = cp.Variable((p+1,p+1), symmetric=True)
    constraints= []
    constraints += [X >> 0]              # The operator >> denotes matrix inequality.
    constraints += [cp.trace(B1 @ X) <= Lmax*n]
    constraints += [cp.trace(B2 @ X) <= -Amin*n]
    constraints += [cp.trace(B3[i] @ X) == 1 for i in range(n)]
    constraints += [cp.trace(B4[i] @ X) <= bmax[i] for i in range(m)]
    constraints += [cp.trace(B5[i] @ X) == 0 for i in range(p)]
    # constraints += [X<= 1, X>= 0]   # Convex Relaxation 0<=x_i,y_{ij}<=1
    # constraints += [ X>= 0]    

    prob = cp.Problem(cp.Minimize(1/n*cp.trace(B0 @ X)),
                    constraints)
    # prob.solve(solver="MOSEK", verbose=True)
    # prob.solve(solver="SCS")
    # prob.solve(solver="MOSEK")
    # prob.solve(solver="GUROBI",verbose=True)
    prob.solve(solver="SCS")
    # Print result.
    # print("The SDR optimal value is", prob.value)
    print("A solution X is")
    # np.set_printoptions(precision=3)

    print (X.value.any() > 0 and X.value.any() < 1)
    # rank = np.linalg.matrix_rank(X.value)
    # print ("Rank of SDR solution is: ", rank)
    # return X.value

    ############# mapping to feasible solution 
    iteration_best = -1
    Xstar = X.value
    Xstar = Xstar[:-1,:-1]
    minimum_obj = 10000000000000 
    solution=[]
    for l in range (iterations):
        ksi = np.random.multivariate_normal(np.zeros(p), Xstar, tol=1e-6)
        # print (ksi)
        xcandidate = 1/ (1 + np.exp(-20*ksi))   # mapping function
        column_to_be_added = np.zeros([n,m+1])
        for i in range (n):
            for j in range (m):
                column_to_be_added[i,0] = xcandidate[i]
                column_to_be_added[i][j+1] = f'{xcandidate[n+m*i+j]:.20f}'
        b = np.zeros_like(column_to_be_added)
        b[np.arange(len(column_to_be_added)), column_to_be_added.argmax(1)] = 1
        pert = np.zeros([n*m+n, 1])
        for i in range(n):
            for j in range(m):
                pert[i][0] = b[i][0]
                pert[n+m*i+j][0] = b[i][j+1]
        pert = np.append(pert,1)
        pert = np.array([pert])
        Y = np.transpose(pert)*pert
        L = np.trace(B1 @ Y)
        A = np.trace(B2 @ Y)
        if L < Lmax*n and A <= -Amin*n and all([np.trace(B3[i] @ Y) == 1 for i in range(n)]) and all ([np.trace(B4[i] @ Y) <= bmax[i] for i in range(m)]):
            candidate = 1/n*np.trace(B0 @ Y)
            if candidate < minimum_obj:
                minimum_obj= candidate
                solution = pert 
                iteration_best = l
                Lbest = L/n
                Amax= -A/n
        # else: 
        #     candidate = 1/n*np.trace(B0 @ Y)
        #     if candidate< minimum_obj:
        #         print ("ring the bell")
        #         print ("solution ", candidate)
        #         print ("solution ", pert)

    # print ("iteration number to find optimal SDR solution: ", iteration_best)
    # print ("minimum objective ", minimum_obj)     
    sdr_solution = solution[0][:-1]
    # print ("solution ", sdr_solution)

    # compute best L and A
    # print ("Lbest: ", Lbest)
    # print  ("Amax: ", Amax)
    return sdr_solution, Lbest, Amax

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

def calculate_cost(solution,B0,B1,B2,B3,B4):
    pert = solution.tolist()
    minimum_obj = 10000000000000 
    pert.append(1)
    pert = np.array([pert])
    Y = np.transpose(pert)*pert
    L = np.trace(B1 @ Y)
    A = np.trace(B2 @ Y)
    candidate = 1/n*np.trace(B0 @ Y)
    minimum_obj= candidate
    solution = pert 
    # print ("L: ", L/n)
    # print  ("A: ", -A/n)
    # print ("solution ", solution)
    # print ("minimum objective ", minimum_obj)
    total_cost = L/n +alpha*A/n  # + operator because A is negative here due to B2 matrix
    # print (total_cost)
    return total_cost,L/n,-A/n

def main():
    sdr_solution = [0]
    slist = random_compression()
    Lcurrent = 1000000
    Acurrent = -10000000
    epsilon = 0.01
    counter = 0 
    while (True):
        # print ("\n new iteration number: ", counter)
        prev_sol = sdr_solution
        B0,B1,B2,B3,B4,B5 = initilization(slist)
        sdr_solution, Lbest, Amax = sdr_offloading(B0,B1,B2,B3,B4,B5) 
        # print (sdr_solution)
        # sdr_solution = random_offloading(B0,B1,B2,B3,B4)
        # sdr_solution, Lbest, Amax  = implement_brute_force(B0,B1,B2,B3,B4,B5)
        # slist = random_compression()
        # print ("Total Cost of Solution: ", calculate_cost(sdr_solution,B0,B1,B2,B3,B4))
        slist = scaling_problem(sdr_solution)
        # if not((sdr_solution == prev_sol).all()) and counter != 0 :
        #     print("different solution found:")
        #     print ("new solution:" , sdr_solution)
        # #     print ("previous solution:" ,prev_sol)
        # if (Lbest-alpha*Amax <= Lcurrent-alpha*Acurrent + epsilon) and (Lbest-alpha*Amax >= Lcurrent-alpha*Acurrent - epsilon):
        #     # print ("condition2")
        #     break; 
        # Lcurrent = Lbest
        # Acurrent = Amax
        counter +=1
        break;
    total_cost, Lfinal,Afinal = calculate_cost(sdr_solution,B0,B1,B2,B3,B4)
    print (total_cost)
    print (Lfinal,Afinal)
    return 

if __name__ == "__main__":
    main()
