import json
import os

def get_project_root():
    """Get the project root directory dynamically"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return current_dir

class SimpleAdditionNetwork:
    def __init__(self, w1, w2, bias):
        self.w1, self.w2, self.bias = w1, w2, bias

    def predict(self, a, b):
        return self.w1 * a + self.w2 * b + self.bias

def interactive_addition(model):
    while True:
        try:
            inp = input("\n\nEnter 'a + b' : ").strip()
            if inp.lower() == 'quit':
                break
            if '+' in inp:
                parts = inp.split('+')
            else:
                parts = inp.split()
            a, b = map(int, (p.strip() for p in parts))
            print(f"\nNeural Network: {a} + {b} = {round(model.predict(a, b))}")
        except (ValueError, KeyboardInterrupt):
            break

if __name__ == "__main__":
    project_root = get_project_root()
    weights_path = os.path.join(project_root, "DATA", "weights.json")
    
    try:
        with open(weights_path) as f:
            weights = json.load(f)
    except FileNotFoundError:
        exit("No DATA/weights.json found! Run train_model.py first which is in the scripts folder.")

    model = SimpleAdditionNetwork(weights["w1"], weights["w2"], weights["bias"])
    print(f"Loaded model: single_layer_nn: w1={model.w1:.6f}, w2={model.w2:.6f}, bias={model.bias:.6f}")
    print("You can now use the model to predict addition problems. Just enter 'a + b' and the model will give you the answer.")
    interactive_addition(model)
