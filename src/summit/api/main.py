import uvicorn

from summit.api import infer

def main():
    uvicorn.run(infer.app, host="0.0.0.0", port=8080, log_level="trace")

if __name__ == "__main__":
    main()