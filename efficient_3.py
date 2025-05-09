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
    time_taken = (end_time - start_time) * 1000
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

    # Solve Min cost first, then backtrack
    min_cost = compute_cost(s1, s2)    
    aligned_s1, aligned_s2 = hirschberg(s1, s2)
    
    return min_cost, aligned_s1, aligned_s2

def compute_cost(s1, s2, isBacktrack=False):

    m, n = len(s1), len(s2)
    
    prev_row = [j * PENALTY for j in range(n+1)]
    curr_row = [0] * (n+1)
    
    for i in range(1, m+1):
        curr_row[0] = i * PENALTY
        
        for j in range(1, n+1):
            match_cost = prev_row[j-1] + MISMATCH_COST.get((s1[i-1], s2[j-1]), 0)
            gap_x_cost = prev_row[j] + PENALTY
            gap_y_cost = curr_row[j-1] + PENALTY
            
            curr_row[j] = min(match_cost, gap_x_cost, gap_y_cost)
        
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[n] if not isBacktrack else prev_row

def hirschberg(s1, s2):

    m, n = len(s1), len(s2)
    
    # Base cases
    if m == 0:
        return '-' * n, s2
    if n == 0:
        return s1, '-' * m
    if m == 1 or n == 1:
        return base(s1, s2)
        
    mid = m // 2
    
    # Compute forward score (from start to mid)
    forward_last_row = compute_cost(s1[:mid], s2, isBacktrack=True)
    
    # Compute backward score (from end to mid, reversed)
    backward_last_row = compute_cost(s1[mid:][::-1], s2[::-1], isBacktrack=True)
    backward_last_row.reverse()
    
    split_idx = 0
    min_score = float('inf')
    for j in range(n+1):
        score = forward_last_row[j] + backward_last_row[j]
        if score < min_score:
            min_score = score
            split_idx = j
    
    s1_left, s2_left = hirschberg(s1[:mid], s2[:split_idx])
    s1_right, s2_right = hirschberg(s1[mid:], s2[split_idx:])

    return s1_left + s1_right, s2_left + s2_right

def base(s1, s2):
    m, n = len(s1), len(s2)
    
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    backtrace = [[None for _ in range(n+1)] for _ in range(m+1)]

    for j in range(n+1):
        dp[0][j] = j * PENALTY
        backtrace[0][j] = (0, j-1) if j > 0 else None
    
    for i in range(m+1):
        dp[i][0] = i * PENALTY
        backtrace[i][0] = (i-1, 0) if i > 0 else None
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            match_cost = dp[i-1][j-1] + MISMATCH_COST.get((s1[i-1], s2[j-1]), 0)
            gap_x_cost = dp[i-1][j] + PENALTY
            gap_y_cost = dp[i][j-1] + PENALTY
            
            if match_cost <= gap_x_cost and match_cost <= gap_y_cost:
                dp[i][j] = match_cost
                backtrace[i][j] = (i-1, j-1) 
            elif gap_x_cost <= gap_y_cost:
                dp[i][j] = gap_x_cost
                backtrace[i][j] = (i-1, j)
            else:
                dp[i][j] = gap_y_cost
                backtrace[i][j] = (i, j-1) 

    aligned_s1 = []
    aligned_s2 = []
    i, j = m, n
    
    while i > 0 or j > 0:
        prev_i, prev_j = backtrace[i][j]
        
        if prev_i == i-1 and prev_j == j-1:
            aligned_s1.append(s1[i-1])
            aligned_s2.append(s2[j-1])
        elif prev_i == i-1:
            aligned_s1.append(s1[i-1])
            aligned_s2.append('-')
        else:
            aligned_s1.append('-')
            aligned_s2.append(s2[j-1])
            
        i, j = prev_i, prev_j
    
    aligned_s1.reverse()
    aligned_s2.reverse()
    
    return ''.join(aligned_s1), ''.join(aligned_s2)

def main():
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    mem = process_memory()
    s1, s2 = read_data(input_file)
    
    result, exec_time = time_wrapper(solve, s1, s2)
    min_cost, aligned_s1, aligned_s2 = result
    mem_used = process_memory() - mem

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
        f.write(str(exec_time))
        f.write('\n')
        f.write(str(mem_used))


if __name__ == "__main__":
    main()