import random, math, os, time, json

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SimpleAdditionNetwork:
    def __init__(self, max_number=1000):
        r = random.uniform
        self.w1, self.w2, self.bias = r(0.5, 1.5), r(0.5, 1.5), r(-0.5, 0.5)
        self.lr = 1.0 / (max_number * max_number)

    def predict(self, a, b):
        return self.w1 * a + self.w2 * b + self.bias

    def train(self, training_data, epochs=10000):
        if not training_data: return
        n, start = len(training_data), time.time()
        
        for epoch in range(epochs):
            total = 0.0
            random.shuffle(training_data)
            for a, b, expected in training_data:
                pred, err = self.predict(a, b), self.predict(a, b) - expected
                if math.isnan(err) or abs(err) > 1e10:
                    self.w1 = self.w2 = 1.0; self.bias = 0.0; self.lr *= 0.1; continue
                total += err * err
                # gradient update with clipping
                clip = lambda x, v: max(-10, min(10, x - self.lr * max(-100, min(100, v))))
                self.w1, self.w2, self.bias = clip(self.w1, err * a), clip(self.w2, err * b), clip(self.bias, err)

            # progress logging
            interval = 10 if epoch < 100 else 50 if epoch < 1000 else 100
            if (epoch + 1) % interval == 0 or epoch == epochs - 1:
                os.system('cls' if os.name == 'nt' else 'clear')
                avg = total / n
                print(f"Epoch {epoch+1:4d} | Loss: {avg:10.6f} | LR: {self.lr:.8f} | w1:{self.w1:.6f} w2:{self.w2:.6f} bias:{self.bias:.6f}")
                progress, bar_len = (epoch + 1) / epochs, 70
                filled = int(bar_len * progress)
                bar = '█' * filled + ' ' * (bar_len - filled)
                elapsed, it_s = time.time() - start, (epoch + 1) / (time.time() - start) if time.time() > start else 0
                eta = (epochs - epoch - 1) / it_s if it_s > 0 else 0
                print(f"{progress*100:3.0f}%|{bar}| {epoch+1}/{epochs} [{int(elapsed//60):02d}:{int(elapsed%60):02d}<{int(eta//60):02d}:{int(eta%60):02d}, {it_s:.2f}it/s]")
                if avg < 1e-10: print(f"Converged at epoch {epoch+1}"); break

def load_addition_dataset():
    dataset_path = os.path.join(get_project_root(), "DATA", "dataset.txt")
    try:
        with open(dataset_path) as f:
            return [(a, b, r) for line in f if line.strip() for a, b, r in [map(int, line.strip().split(","))] if a + b == r]
    except (FileNotFoundError, ValueError):
        print("Creating sample dataset...")
        return [(a := random.randint(1, 100), b := random.randint(1, 100), a + b) for _ in range(1000)]

def test_accuracy(model, test_cases):
    return sum(1 for a, b, expected in test_cases if abs(round(model.predict(a, b)) - expected) <= 1) / len(test_cases) * 100 if test_cases else 0.0

if __name__ == "__main__":
    training_data = load_addition_dataset()
    if not training_data: exit("No valid data available")

    print(f"Training with {len(training_data)} examples")
    max_input = max(max(a, b) for a, b, _ in training_data)
    model = SimpleAdditionNetwork(max_input)
    print(f"Initial weights: w1={model.w1:.6f}, w2={model.w2:.6f}, bias={model.bias:.6f}")

    model.train(training_data)

    print(f"Final weights: w1={model.w1:.6f}, w2={model.w2:.6f}, bias={model.bias:.6f}")
    print(f"Training accuracy: {test_accuracy(model, training_data[:100]):.0f}%")

    # save weights
    weights_path = os.path.join(get_project_root(), "DATA", "weights.json")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    with open(weights_path, "w") as f:
        json.dump({"w1": model.w1, "w2": model.w2, "bias": model.bias}, f, indent=2)
    print("\nModel weights saved to DATA/weights.json")
