from pagerank import *

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    #model = transition_model(corpus, "2.html", DAMPING)
    sample_pagerank(corpus, DAMPING, SAMPLES)


if __name__ == "__main__":
    main()
