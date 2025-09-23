import random
import os

def get_project_root():
    """Get the project root directory dynamically"""
    # Get the directory of this script, then go up one level to project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def generate_addition_dataset():
    """Generate a comprehensive addition dataset"""
    problems = []
    
    print("Generating addition problems for dataset.txt...")
    
    # 1. Small numbers (1-20) - comprehensive coverage
    print("  Adding small numbers (1-20)...")
    for a in range(1, 21):
        for b in range(1, 21):
            problems.append((a, b, a + b))
    
    # 2. Medium numbers with variety
    print("  Adding medium numbers (21-100)...")
    for _ in range(200):
        a = random.randint(21, 100)
        b = random.randint(21, 100)
        problems.append((a, b, a + b))
    
    # 3. Large numbers for generalization testing
    print("  Adding large numbers (101-500)...")
    for _ in range(100):
        a = random.randint(101, 500)
        b = random.randint(101, 500)
        problems.append((a, b, a + b))
    
    
    # Powers of 2
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128]
    for a in powers_of_2:
        for b in powers_of_2:
            if a + b <= 1000:
                problems.append((a, b, a + b))
    
    # Multiples of 5 and 10
    multiples_5 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for a in multiples_5:
        for b in multiples_5:
            problems.append((a, b, a + b))
    
    # Same number additions (doubles)
    for i in range(1, 51):
        problems.append((i, i, i + i))
    
    # Adding with 0 and 1
    for i in range(1, 21):
        problems.append((i, 0, i))
        problems.append((0, i, i))
        problems.append((i, 1, i + 1))
        problems.append((1, i, 1 + i))
    
    # 5. Edge cases for robustness
    print("  Adding edge cases...")
    edge_cases = [
        (999, 1, 1000),
        (500, 500, 1000),
        (123, 456, 579),
        (789, 211, 1000),
        (99, 99, 198)
    ]
    problems.extend(edge_cases)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_problems = []
    for problem in problems:
        if problem not in seen:
            seen.add(problem)
            unique_problems.append(problem)
    
    # Shuffle for better training
    random.shuffle(unique_problems)
    
    return unique_problems

def main():
    print("ADDITION DATASET GENERATOR")
    print("=" * 40)
    
    # Generate the dataset
    dataset = generate_addition_dataset()
    
    # Write to file
    project_root = get_project_root()
    dataset_path = os.path.join(project_root, "DATA", "dataset.txt")
    
    # Ensure DATA directory exists
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    with open(dataset_path, "w") as f:
        for a, b, result in dataset:
            f.write(f"{a},{b},{result}\n")
    
    print(f"\nGenerated dataset.txt with {len(dataset)} problems.")
    
    # Show statistics
    print("\nDATASET STATISTICS:")
    print(f"  Total problems:     {len(dataset)}")
    
    max_a = max(a for a, b, _ in dataset)
    max_b = max(b for a, b, _ in dataset)
    max_result = max(result for _, _, result in dataset)
    
    print(f"  Max first number:   {max_a}")
    print(f"  Max second number:  {max_b}")
    print(f"  Max result:         {max_result}")
    
    # Show some examples
    print("\nSample problems:")
    for i in range(10):
        a, b, result = dataset[i]
        print(f"  {a} + {b} = {result}")
    
    print("\nDataset is ready for training.")

if __name__ == "__main__":
    main()
