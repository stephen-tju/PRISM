import requests
import time

targets = [
    "http://localhost:8000/metrics",
    "http://localhost:8010/metrics",
    "http://localhost:8002/metrics",
    "http://localhost:8003/metrics",
    # "http://localhost:8004/metrics",
    # "http://localhost:8005/metrics",
    # "http://localhost:8006/metrics",
    # "http://localhost:8007/metrics",
]

n = 15  

def check_all_targets():
    for target in targets:
        try:
            response = requests.get(target, timeout=n)  
            response.raise_for_status()  
        except requests.RequestException as e:
            print(f" {target}: {e}")
            return False
    return True

def main():
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  checking target address ...")
    while True:
        if check_all_targets():
            print("All addresses are reachable.")
            break
        else:
            print(f"Waiting for {n} seconds before retrying...")
            time.sleep(n)

if __name__ == "__main__":
    main()
