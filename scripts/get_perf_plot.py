from torchstate.server import StateServer
import time
import torch

def bench(N: int) -> float:
    state_dict = {
        "tensor": torch.randn(N)
    }

    server = StateServer(state_dict)
    server.start()

    # CLIENT BEGIN
    from torchstate.client import StateClient

    client = StateClient("localhost:12345")

    time_takens = []

    for _ in range(10):
        start_time = time.perf_counter()
        a = client.get_tensor("tensor")
        time_takens.append(time.perf_counter() - start_time)

    mbps = a.element_size() * a.numel() / (1024**2) / (sum(time_takens) / len(time_takens))
    print(f"MB/s: {mbps}")
    # CLIENT END

    time.sleep(1)
    server.stop()

    return mbps

if __name__ == "__main__":
    aa = []
    bb = []
    for i in range(8):
        a = 10**i
        b = bench(a)
        aa.append(a)
        bb.append(b)
    
    from matplotlib import pyplot as plt
    plt.plot(aa, bb)
    plt.xscale('log')
    plt.savefig('meow.png')
    plt.title("Bandwidth utilization vs tensor size")
    plt.xlabel("Tensor size")
    plt.ylabel("MB/s")