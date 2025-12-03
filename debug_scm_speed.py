import time
from src.data.scm_generator import SCMGenerator

def test_speed():
    start = time.time()
    scm = SCMGenerator(num_vars=100, edge_prob=0.1)
    print(f"Init time: {time.time() - start:.4f}s")
    
    start = time.time()
    data = scm.generate_data(n_samples=100)
    print(f"Generate time: {time.time() - start:.4f}s")
    print(f"Data shape: {data.shape}")

if __name__ == "__main__":
    test_speed()
