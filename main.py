import sys
import sys
from resources import *
import time
import psutil

# ------------- Utils -------------------
def process_memory() :
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_consumed = int(memory_info.rss / 1024)
    return memory_consumed

def time_wrapper():
    start_time = time.time()
    # call_algorithm() # Replace with your algorithm function call
    end_time = time.time()
    time_taken = (end_time - start_time ) * 1000
    return time_taken

def read_data(input_file):
    with open(input_file, 'r') as f:
        s1, s2 = "", ""
        for line in f:
            line = line.strip()
            if line.isnumeric():
                l = int(line)
                if s2 == "":
                    s1 = s1[:l+1] + s1 + s1[l+1:]
                else:
                    s2 = s2[:l+1] + s2 + s2[l+1:]
            else:
                if s1 == "":
                    s1 = line
                else:
                    s2 = line

    return s1, s2

def optimal_alignment(X, Y, gap_penalty, mismatch_cost):
    """
    Find the optimal alignment between two strings X and Y.
    
    Args:
        X (str): First string
        Y (str): Second string
        gap_penalty (float): Cost for each gap (δ)
        mismatch_cost (dict): Dictionary mapping pairs of characters to their mismatch costs
        
    Returns:
        tuple: (minimum cost, alignment of X, alignment of Y)
    """
    m, n = len(X), len(Y)
    
    # Initialize the DP table with zeros
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    # Initialize the backtrace matrix to reconstruct the alignment
    backtrace = [[None for _ in range(n+1)] for _ in range(m+1)]
    
    # Fill in the first row (aligning Y with gaps)
    for j in range(1, n+1):
        dp[0][j] = j * gap_penalty
        backtrace[0][j] = (0, j-1)
    
    # Fill in the first column (aligning X with gaps)
    for i in range(1, m+1):
        dp[i][0] = i * gap_penalty
        backtrace[i][0] = (i-1, 0)
    
    # Fill the rest of the DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Calculate costs for the three possible moves
            match_cost = dp[i-1][j-1] + mismatch_cost.get((X[i-1], Y[j-1]), 0)
            gap_x_cost = dp[i-1][j] + gap_penalty
            gap_y_cost = dp[i][j-1] + gap_penalty
            
            # Choose the minimum cost action
            if match_cost <= gap_x_cost and match_cost <= gap_y_cost:
                dp[i][j] = match_cost
                backtrace[i][j] = (i-1, j-1)  # diagonal move (match/mismatch)
            elif gap_x_cost <= gap_y_cost:
                dp[i][j] = gap_x_cost
                backtrace[i][j] = (i-1, j)    # up move (gap in Y)
            else:
                dp[i][j] = gap_y_cost
                backtrace[i][j] = (i, j-1)    # left move (gap in X)
    
    # Reconstruct the alignment using the backtrace matrix
    aligned_X = []
    aligned_Y = []
    i, j = m, n
    
    while i > 0 or j > 0:
        prev_i, prev_j = backtrace[i][j]
        
        if prev_i == i-1 and prev_j == j-1:
            # Match or mismatch
            aligned_X.append(X[i-1])
            aligned_Y.append(Y[j-1])
        elif prev_i == i-1:
            # Gap in Y
            aligned_X.append(X[i-1])
            aligned_Y.append('-')
        else:
            # Gap in X
            aligned_X.append('-')
            aligned_Y.append(Y[j-1])
            
        i, j = prev_i, prev_j
    
    # Reverse the alignments (we built them backwards)
    aligned_X.reverse()
    aligned_Y.reverse()
    
    return dp[m][n], ''.join(aligned_X), ''.join(aligned_Y)

def main():
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    s1, s2 = read_data(input_file)
    print(s1)
    print(s2)
    
    # Mismatch costs
    gap_penalty = 30
    mismatch_cost = {
        ('A', 'A'): 0, ('C', 'C'): 0, ('G', 'G'): 0, ('T', 'T'): 0,
        
        ('A', 'G'): 48, ('G', 'A'): 48, 
        ('C', 'T'): 48, ('T', 'C'): 48,
        ('A', 'C'): 110, ('C', 'A'): 110,
        ('A', 'T'): 94, ('T', 'A'): 94,
        ('G', 'C'): 118, ('C', 'G'): 118,
        ('G', 'T'): 110, ('T', 'G'): 110,
    }
    
    min_cost, aligned_X, aligned_Y = optimal_alignment(s1, s2, gap_penalty, mismatch_cost)
    
    print(f"Minimum alignment cost: {min_cost}")
    print(f"Aligned X: {aligned_X}")
    print(f"Aligned Y: {aligned_Y}")
    
    # Print the alignment in a more readable format
    print("\nAlignment:")
    for x, y in zip(aligned_X, aligned_Y):
        if x == y:
            print("|", end="")
        elif x == '-' or y == '-':
            print(" ", end="")
        else:
            print(".", end="")
    print()
    print(aligned_X)
    print(aligned_Y)

if __name__ == "__main__":
    main()