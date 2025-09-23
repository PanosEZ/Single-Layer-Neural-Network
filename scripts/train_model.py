import random, math, os, time, json

def get_project_root():
    """Get the project root directory dynamically"""
    # Get the directory of this script, then go up one level to project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

class SimpleAdditionNetwork:
    def __init__(self, max_number=1000):
        r = random.uniform
        self.w1, self.w2 = r(0.5, 1.5), r(0.5, 1.5)
        self.bias = r(-0.5, 0.5)
        self.lr = 1.0 / (max_number * max_number)

    def predict(self, a, b):
        return self.w1 * a + self.w2 * b + self.bias

    def train(self, training_data, epochs=10000):
        if not training_data:
            return
        n = len(training_data)
        start = time.time()
        for epoch in range(epochs):
            total = 0.0
            random.shuffle(training_data)
            for a, b, expected in training_data:
                pred = self.predict(a, b)
                err = pred - expected
                if math.isnan(err) or abs(err) > 1e10:
                    self.w1 = self.w2 = 1.0
                    self.bias = 0.0
                    self.lr *= 0.1
                    continue
                total += err * err
                # gradient update with clipping
                self.w1 = max(-10, min(10, self.w1 - self.lr * max(-100, min(100, err * a))))
                self.w2 = max(-10, min(10, self.w2 - self.lr * max(-100, min(100, err * b))))
                self.bias = max(-10, min(10, self.bias - self.lr * max(-100, min(100, err))))

            # progress logging
            interval = 10 if epoch < 100 else 50 if epoch < 1000 else 100
            if (epoch + 1) % interval == 0 or epoch == epochs - 1:
                os.system('cls' if os.name == 'nt' else 'clear')
                avg = total / n
                print(f"Epoch {epoch+1:4d} | Loss: {avg:10.6f} | LR: {self.lr:.8f} | "
                      f"w1:{self.w1:.6f} w2:{self.w2:.6f} bias:{self.bias:.6f}")
                progress = (epoch + 1) / epochs
                bar_len = 70
                filled = int(bar_len * progress)
                bar = '█' * filled + ' ' * (bar_len - filled)
                elapsed = time.time() - start
                it_s = (epoch + 1) / elapsed if elapsed > 0 else 0
                eta = (epochs - epoch - 1) / it_s if it_s > 0 else 0
                print(f"{progress*100:3.0f}%|{bar}| {epoch+1}/{epochs} "
                      f"[{int(elapsed//60):02d}:{int(elapsed%60):02d}<{int(eta//60):02d}:{int(eta%60):02d}, {it_s:.2f}it/s]")
                if avg < 1e-10:
                    print(f"Converged at epoch {epoch+1}")
                    break

def load_addition_dataset():
    project_root = get_project_root()
    dataset_path = os.path.join(project_root, "DATA", "dataset.txt")
    
    try:
        with open(dataset_path) as f:
            out = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    a, b, r = map(int, line.split(","))
                    if a + b == r:
                        out.append((a, b, r))
                except ValueError:
                    pass
            return out
    except FileNotFoundError:
        print("Creating sample dataset...")
        out = []
        for _ in range(1000):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            out.append((a, b, a + b))
        return out

def test_accuracy(model, test_cases):
    correct = sum(1 for a, b, expected in test_cases if abs(round(model.predict(a, b)) - expected) <= 1)
    return correct / len(test_cases) * 100 if test_cases else 0.0

if __name__ == "__main__":
    training_data = load_addition_dataset()
    if not training_data:
        exit("No valid data available")

    print(f"Training with {len(training_data)} examples")
    max_input = max(max(a, b) for a, b, _ in training_data)
    model = SimpleAdditionNetwork(max_input)
    print(f"Initial weights: w1={model.w1:.6f}, w2={model.w2:.6f}, bias={model.bias:.6f}")

    model.train(training_data)

    print(f"Final weights: w1={model.w1:.6f}, w2={model.w2:.6f}, bias={model.bias:.6f}")
    print(f"Training accuracy: {test_accuracy(model, training_data[:100]):.0f}%")

    # save weights to file
    project_root = get_project_root()
    weights_path = os.path.join(project_root, "DATA", "weights.json")
    
    # Ensure DATA directory exists
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    
    weights = {"w1": model.w1, "w2": model.w2, "bias": model.bias}
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)

    print("\nModel weights saved to DATA/weights.json")
