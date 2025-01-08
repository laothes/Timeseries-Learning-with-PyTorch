import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    parser = argparse.ArgumentParser(description="Example parser script.")
    parser.add_argument("--name", type=str, help="Your name", required=True)
    parser.add_argument("--age", type=int, help="Your age", required=False)
    args = parser.parse_args()

    logging.info(f"Hello, {args.name}!")
    if args.age:
        logging.info(f"You are {args.age} years old.")

if __name__ == "__main__":
    main()