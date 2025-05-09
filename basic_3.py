import sys
import os
from resources import *
import time
import psutil

# -------- GLOBAL VARIABLES -------------
PENALTY = 30
MISMATCH_COST = {
    ('A', 'A'): 0, ('C', 'C'): 0,
    ('G', 'G'): 0, ('T', 'T'): 0,
    ('A', 'G'): 48, ('G', 'A'): 48, 
    ('C', 'T'): 48, ('T', 'C'): 48,
    ('A', 'C'): 110, ('C', 'A'): 110,
    ('A', 'T'): 94, ('T', 'A'): 94,
    ('G', 'C'): 118, ('C', 'G'): 118,
    ('G', 'T'): 110, ('T', 'G'): 110,
}

# ------------- Utils -------------------
def process_memory() :
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_consumed = int(memory_info.rss / 1024)
    return memory_consumed

def time_wrapper(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    time_taken = (end_time - start_time ) * 1000
    return result, time_taken

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

# ---------------- Algorithm --------------
def solve(s1, s2):
    m, n = len(s1), len(s2)
    
    # Initialize the DP table
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    backtrack = [[None for _ in range(n+1)] for _ in range(m+1)]

    for j in range(1, n+1):
        dp[0][j] = j * PENALTY
        backtrack[0][j] = (0, j-1)
    
    for i in range(1, m+1):
        dp[i][0] = i * PENALTY
        backtrack[i][0] = (i-1, 0)
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            match_cost = dp[i-1][j-1] + MISMATCH_COST.get((s1[i-1], s2[j-1]), 0)
            gapX_cost = dp[i-1][j] + PENALTY
            gapY_cost = dp[i][j-1] + PENALTY
            
            # Choose the minimum cost action
            if match_cost <= gapX_cost and match_cost <= gapY_cost:
                dp[i][j] = match_cost
                backtrack[i][j] = (i-1, j-1)
            elif gapX_cost <= gapY_cost:
                dp[i][j] = gapX_cost
                backtrack[i][j] = (i-1, j)
            else:
                dp[i][j] = gapY_cost
                backtrack[i][j] = (i, j-1)
    
    aligned_s1 = []
    aligned_s2 = []
    i, j = m, n
    
    while i > 0 or j > 0:
        prev_i, prev_j = backtrack[i][j]
        
        if prev_i == i-1 and prev_j == j-1:
            # Match or mismatch
            aligned_s1.append(s1[i-1])
            aligned_s2.append(s2[j-1])
        elif prev_i == i-1:
            # Gap in Y
            aligned_s1.append(s1[i-1])
            aligned_s2.append('-')
        else:
            # Gap in X
            aligned_s1.append('-')
            aligned_s2.append(s2[j-1])
            
        i, j = prev_i, prev_j
    
    aligned_s1.reverse()
    aligned_s2.reverse()
    
    return dp[m][n], ''.join(aligned_s1), ''.join(aligned_s2)

def main():
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    mem = process_memory()
    s1, s2 = read_data(input_file)
    
    result, execTime = time_wrapper(solve, s1, s2)

    min_cost, aligned_s1, aligned_s2 = result
    memUsed = process_memory() - mem
    
    dir = os.path.dirname(output_file)
    if dir:
        os.makedirs(dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(str(min_cost))
        f.write('\n')
        f.write(aligned_s1)
        f.write('\n')
        f.write(aligned_s2)
        f.write('\n')
        f.write(str(execTime))
        f.write('\n')
        f.write(str(memUsed))


if __name__ == "__main__":
    main()