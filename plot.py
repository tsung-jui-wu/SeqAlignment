#!/usr/bin/env python3
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import re

# File paths for the two implementations
NORMAL_DP_SCRIPT = "basic_3.py"  # Assuming this is your base file
EFFICIENT_DP_SCRIPT = "efficient_3.py"  # Your memory efficient implementation

def get_file_size(file_path):
    """Get the size of input problem by reading the file and calculating final sequence lengths"""
    # Use the read_data function logic to calculate actual sequence sizes
    s1, s2 = read_data(file_path)
    return len(s1) + len(s2)

def read_data(input_file):
    """Read and expand the sequences according to the pattern in the input file"""
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

def run_algorithm(script_path, input_file, output_file):
    """Run the specified algorithm script on the input file and return time and memory usage"""
    result = subprocess.run(
        [sys.executable, script_path, input_file, output_file],
        capture_output=True,
        text=True
    )
    
    # Read time and memory from output file
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            # Based on your file format where the last two lines are time and memory
            if len(lines) >= 5:  # Accounting for cost value and aligned sequences
                time_used = float(lines[-2].strip())
                memory_used = int(lines[-1].strip())
                return time_used, memory_used
    except Exception as e:
        print(f"Error reading output file: {e}")
    
    # Return default values if we couldn't read the output properly
    return 0, 0

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Check if script files exist
    if not os.path.isfile(NORMAL_DP_SCRIPT):
        print(f"Error: Normal DP script '{NORMAL_DP_SCRIPT}' not found.")
        sys.exit(1)
    
    if not os.path.isfile(EFFICIENT_DP_SCRIPT):
        print(f"Error: Efficient DP script '{EFFICIENT_DP_SCRIPT}' not found.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Data structures to store results
    problem_sizes = []
    normal_times = []
    normal_memories = []
    efficient_times = []
    efficient_memories = []
    
    # Process each input file
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    if not input_files:
        print(f"No input files found in directory: {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(input_files)} input files to process.")
    
    for idx, input_file in enumerate(input_files, 1):
        input_path = os.path.join(input_dir, input_file)
        try:
            print(f"\nProcessing file {idx}/{len(input_files)}: {input_file}")
            problem_size = get_file_size(input_path)
            print(f"Calculated problem size: {problem_size} characters")
            problem_sizes.append(problem_size)
            
            # Run normal DP algorithm
            print("Running normal DP algorithm...")
            normal_output = os.path.join(output_dir, f"normal_{input_file}_output.txt")
            normal_time, normal_memory = run_algorithm(NORMAL_DP_SCRIPT, input_path, normal_output)
            normal_times.append(normal_time)
            normal_memories.append(normal_memory)
            
            # Run memory-efficient algorithm
            print("Running memory-efficient DP algorithm...")
            efficient_output = os.path.join(output_dir, f"efficient_{input_file}_output.txt")
            efficient_time, efficient_memory = run_algorithm(EFFICIENT_DP_SCRIPT, input_path, efficient_output)
            efficient_times.append(efficient_time)
            efficient_memories.append(efficient_memory)
            
            print(f"Results for {input_file} (size: {problem_size}):")
            print(f"  Normal DP: Time = {normal_time:.2f}ms, Memory = {normal_memory} KB")
            print(f"  Efficient: Time = {efficient_time:.2f}ms, Memory = {efficient_memory} KB")
            
            # Print improvement metrics
            time_improvement = (normal_time - efficient_time) / normal_time * 100 if normal_time > 0 else 0
            memory_improvement = (normal_memory - efficient_memory) / normal_memory * 100 if normal_memory > 0 else 0
            print(f"  Improvements: Time = {time_improvement:.2f}%, Memory = {memory_improvement:.2f}%")
            
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            # Continue with next file instead of stopping completely
            continue
    
    # Filter out any data points with zero problem size or missing metrics
    valid_indices = []
    for i in range(len(problem_sizes)):
        if (problem_sizes[i] > 0 and 
            normal_times[i] > 0 and efficient_times[i] > 0 and 
            normal_memories[i] > 0 and efficient_memories[i] > 0):
            valid_indices.append(i)
    
    if not valid_indices:
        print("No valid data points to plot. Check if algorithms ran successfully.")
        sys.exit(1)
    
    problem_sizes = [problem_sizes[i] for i in valid_indices]
    normal_times = [normal_times[i] for i in valid_indices]
    normal_memories = [normal_memories[i] for i in valid_indices]
    efficient_times = [efficient_times[i] for i in valid_indices]
    efficient_memories = [efficient_memories[i] for i in valid_indices]
    
    # Sort data points by problem size
    sorted_indices = np.argsort(problem_sizes)
    problem_sizes = [problem_sizes[i] for i in sorted_indices]
    normal_times = [normal_times[i] for i in sorted_indices]
    normal_memories = [normal_memories[i] for i in sorted_indices]
    efficient_times = [efficient_times[i] for i in sorted_indices]
    efficient_memories = [efficient_memories[i] for i in sorted_indices]
    
    print(f"\nGenerating plots for {len(problem_sizes)} valid data points.")
    
    # Generate time vs problem size plot
    plt.figure(figsize=(12, 7))
    plt.plot(problem_sizes, normal_times, 'o-', color='blue', label='Normal DP - O(m*n)')
    plt.plot(problem_sizes, efficient_times, 's-', color='red', label='Memory-Efficient DP - O(m*n)')
    plt.xlabel('Problem Size (Length of Seq1 + Seq2)', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title('Execution Time vs Problem Size', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data point annotations
    for i, (size, time1, time2) in enumerate(zip(problem_sizes, normal_times, efficient_times)):
        plt.annotate(f'{size}', (size, time1), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # Use logarithmic scale if range is large (more than 10x difference)
    if max(normal_times + efficient_times) / max(1, min(normal_times + efficient_times)) > 10:
        plt.yscale('log')
        plt.ylabel('Execution Time (ms) - Log Scale', fontsize=12)
    
    plt.tight_layout()
    time_plot_path = os.path.join(output_dir, 'time_comparison-log.png')
    plt.savefig(time_plot_path, dpi=300)
    
    # Generate memory vs problem size plot
    plt.figure(figsize=(12, 7))
    plt.plot(problem_sizes, normal_memories, 'o-', color='blue', label='Normal DP - O(m*n)')
    plt.plot(problem_sizes, efficient_memories, 's-', color='red', label='Memory-Efficient DP - O(m)')
    plt.xlabel('Problem Size (Length of Seq1 + Seq2)', fontsize=12)
    plt.ylabel('Memory Usage (KB)', fontsize=12)
    plt.title('Memory Usage vs Problem Size', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data point annotations
    for i, (size, mem1, mem2) in enumerate(zip(problem_sizes, normal_memories, efficient_memories)):
        plt.annotate(f'{size}', (size, mem1), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # Use logarithmic scale if range is large (more than 10x difference)
    if max(normal_memories + efficient_memories) / max(1, min(normal_memories + efficient_memories)) > 10:
        plt.yscale('log')
        plt.ylabel('Memory Usage (KB) - Log Scale', fontsize=12)
    
    plt.tight_layout()
    memory_plot_path = os.path.join(output_dir, 'memory_comparison-log.png')
    plt.savefig(memory_plot_path, dpi=300)
    
    # Save the raw data as CSV for further analysis
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    with open(csv_path, 'w') as f:
        f.write('problem_size,normal_time_ms,normal_memory_kb,efficient_time_ms,efficient_memory_kb,time_improvement_pct,memory_improvement_pct\n')
        for i in range(len(problem_sizes)):
            time_imp = (normal_times[i] - efficient_times[i]) / normal_times[i] * 100 if normal_times[i] > 0 else 0
            mem_imp = (normal_memories[i] - efficient_memories[i]) / normal_memories[i] * 100 if normal_memories[i] > 0 else 0
            f.write(f"{problem_sizes[i]},{normal_times[i]:.2f},{normal_memories[i]},{efficient_times[i]:.2f},{efficient_memories[i]},{time_imp:.2f},{mem_imp:.2f}\n")
    
    # Create a summary for average improvements
    avg_time_imp = sum([(normal_times[i] - efficient_times[i]) / normal_times[i] * 100 if normal_times[i] > 0 else 0 
                         for i in range(len(problem_sizes))]) / len(problem_sizes)
    avg_mem_imp = sum([(normal_memories[i] - efficient_memories[i]) / normal_memories[i] * 100 if normal_memories[i] > 0 else 0 
                        for i in range(len(problem_sizes))]) / len(problem_sizes)
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Benchmark Summary\n")
        f.write(f"----------------\n\n")
        f.write(f"Number of test cases: {len(problem_sizes)}\n")
        f.write(f"Problem size range: {min(problem_sizes)} to {max(problem_sizes)} characters\n\n")
        f.write(f"Average time improvement: {avg_time_imp:.2f}%\n")
        f.write(f"Average memory improvement: {avg_mem_imp:.2f}%\n\n")
        f.write(f"Best time improvement: {max([(normal_times[i] - efficient_times[i]) / normal_times[i] * 100 if normal_times[i] > 0 else 0 for i in range(len(problem_sizes))]):.2f}%\n")
        f.write(f"Best memory improvement: {max([(normal_memories[i] - efficient_memories[i]) / normal_memories[i] * 100 if normal_memories[i] > 0 else 0 for i in range(len(problem_sizes))]):.2f}%\n")
    
    print(f"\nResults saved to {output_dir}")
    print(f"- Time comparison plot: {time_plot_path}")
    print(f"- Memory comparison plot: {memory_plot_path}")
    print(f"- Raw data: {csv_path}")
    print(f"- Summary report: {summary_path}")
    print("\nSummary:")
    print(f"- Average time improvement: {avg_time_imp:.2f}%")
    print(f"- Average memory improvement: {avg_mem_imp:.2f}%")

if __name__ == "__main__":
    main()