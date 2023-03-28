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
    result_dict = dict()

    # If no linked pages, equal distribution. Otherwise assign probability manually
    if len(linked_pages) == 0:
        for key in corpus.keys():
            result_dict[key] = 1/len(corpus.keys())
        return result_dict
    else:
        starting_percent = 1/len(linked_pages)
    
    # Adds an equally likely percent for each linked page (times the damping factor) and then adds the remainder percent split evenly (including given page)
    random_percent = (1-damping_factor)/(len(linked_pages) + 1)
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
    pagerank_dict = dict.fromkeys(corpus.keys(), 1/(len(corpus.keys())))
    test_dict = pagerank_dict.copy()
    while True:
        # Check if dictionary values between N and N+1 differ by more than .001 (if not, break while loop)
        is_done = False  
        for key,value in pagerank_dict.items():        
            if abs(value - test_dict[key]) >= .001 or value == test_dict[key]:    
                test_dict = pagerank_dict.copy()
                is_done = False
                break  
            is_done = True 

        if is_done:
            break
        
        # Calculates pagerank for each main page
        for main_page in pagerank_dict.keys():
            # Initial random assigned probability
            current_pagerank = (1-damping_factor)/len(corpus.keys())
            
            # Creates a list of all pages that link to the main page
            pages_linking_to = []
            for sub_page in pagerank_dict.keys():
                if len(corpus[sub_page]) == 0:
                    pages_linking_to.append(sub_page)
                if main_page in corpus[sub_page]:
                    pages_linking_to.append(sub_page)
            
            # Summation set up based on pageranks of all pages that link to the main page (multiplied by damping_factor)
            secondary_value = 0 
            for page in pages_linking_to:
                numlinks = len(corpus[page])
                if numlinks == 0: numlinks = len(corpus.keys())
                secondary_value += (pagerank_dict[page]/numlinks)
            secondary_value *= damping_factor
            pagerank_dict[main_page] = current_pagerank + secondary_value
    return pagerank_dict

    
if __name__ == "__main__":
    main()
