from pagerank import *

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    transition_model(corpus, "2.html", DAMPING)

if __name__ == "__main__":
    main()
