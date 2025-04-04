import numpy as np

def lu_factorization(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.copy(A).astype(float)
    
    # Set diagonal elements of L to 1
    for i in range(n):
        L[i, i] = 1.0
    
    for i in range(n):
        for j in range(i+1, n):
            if U[i, i] == 0:
                continue  # Skip if pivot is zero to avoid division by zero
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
    
    return L, U

def diagonally_dominant(A):
    for i in range(len(A)):
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(len(A)) if j != i):
            return False
    return True

def positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def main():
    # Original matrix
    A = np.array([
        [2, -1, 1],
        [1, 3, 1],
        [-1, 5, 4]
    ], dtype=float)
    b = np.array([6, 0, -3], dtype=float)
    
    factor1 = A[1,0] / A[0,0]
    factor2 = A[2,0] / A[0,0]
    
    A_new = A.copy()
    b_new = b.copy()

    A_new[1, :] = A[1, :] - factor1 * A[0, :]
    b_new[1] = b[1] - factor1 * b[0]
    
    A_new[2, :] = A[2, :] - factor2 * A[0, :]
    b_new[2] = b[2] - factor2 * b[0]
    factor3 = A_new[2,1] / A_new[1,1]

    print(f"{factor1}") 
    print(f"{factor3}")
    
    A_copy = A.copy()
    b_copy = b.copy()
    n = len(b)
    
    for k in range(n-1):
        for i in range(k+1, n):
            factor = A_copy[i,k] / A_copy[k,k]
            A_copy[i,k:] = A_copy[i,k:] - factor * A_copy[k,k:]
            b_copy[i] = b_copy[i] - factor * b_copy[k]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b_copy[i] - np.sum(A_copy[i,i+1:] * x[i+1:])) / A_copy[i,i]
    
    print(f"[{int(x[0])} {int(x[1])} {int(x[2])}]")

    # Problem 2 setup
    A2 = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ], dtype=float)
    
    print("\n")
    # Calculating the determinant
    det = np.linalg.det(A2)
    print(f"{det}")
    
    # Calculating LU factorization
    L, U = lu_factorization(A2)
    
    # Printing L matrix
    print("\n")
    print("[[", end="")
    for i in range(4):
        if i > 0:
            print(" [", end="")
        for j in range(4):
            print(f"{L[i, j]:g}.", end="")
            if j < 3:
                print(" ", end="")
        if i < 3:
            print("]")
        else:
            print("]]")
    
    # Printing U matrix
    print("\n")
    print("[[", end="")
    for i in range(4):
        if i > 0:
            print(" [", end="")
        for j in range(4):
            print(f"{U[i, j]:g}.", end="")
            if j < 3:
                print(" ", end="")
        if i < 3:
            print("]")
        else:
            print("]]")

    # Problem 3
    A3 = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]])
    print(diagonally_dominant(A3))

    # Problem 4
    A4 = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]])
    
    print(positive_definite(A4))

    
if __name__ == "__main__":
    main()