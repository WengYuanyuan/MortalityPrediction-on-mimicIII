from collections import Counter
import csv

def vectorize_text_feature(text):
    with open('categories.csv', 'r') as infile:
        reader = csv.reader(infile)
        list_of_categories = [row for row in reader]
    result = []
    c = Counter(text.split())
    for category in list_of_categories:
        total = 0
        for word in category:
            total += c[word]
        result.append(total)
    return result

