import nltk

def download_nltk_resources():
    """Download required NLTK resources."""
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'punkt_tab'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)

if __name__ == "__main__":
    download_nltk_resources()
    print("All NLTK resources have been downloaded successfully!") 