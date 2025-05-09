import sys

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

def read_file(input_file):
    with open(input_file, 'r') as f:
        l = []
        for line in f:
            line = line.strip()
            l.append(line)

    return l

def main():
    
    input_file = sys.argv[1]

    l = read_file(input_file)
    final = l[0]
    score = 0
    for i in range(len(l[1])):
        if l[1][i] == l[2][i]:
            continue
        elif l[1][i] == '-' or l[2][i] == '-':
            score += PENALTY
        else:
            score += MISMATCH_COST.get((l[1][i], l[2][i]), 0)

    print(f"Score: {final}; Matching: {score}")

if __name__ == '__main__':
    main()