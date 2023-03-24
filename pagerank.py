import os
import random
import re
import sys
from unittest import result

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    linked_pages = corpus[page]
    starting_percent = 1/len(linked_pages)
    random_percent = (1-damping_factor)/(len(linked_pages) + 1)
    result_dict = dict()
    for linked_page in linked_pages:
        result_dict[linked_page] = starting_percent * damping_factor
        result_dict[linked_page] += random_percent

    result_dict[page] = random_percent

    return result_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    freq_dict = dict.fromkeys(corpus.keys(), 0)
    
    # Starts initial sample randomly
    current_page = None
    if not current_page:
        current_page = random.choice(list(corpus.keys()))
        freq_dict[current_page] += 1
    
    # Runs n-1 samples based of current_page's transition model each time
    for sample in range(n-1):
        current_transition = transition_model(corpus, current_page, damping_factor)
        random_sample = (random.choices(list(current_transition.keys()), weights=current_transition.values(), k=1))[0]
        current_page = random_sample
        freq_dict[random_sample] += 1

    pagerank_dict = dict.fromkeys(corpus.keys(), 0)
    # Dictionary comprehension makes new dictionary with ratios of sample numbers for pagerank values
    pagerank_dict = {key:(freq_dict[key])/SAMPLES for (key, value) in freq_dict.items()}
    
    return pagerank_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
