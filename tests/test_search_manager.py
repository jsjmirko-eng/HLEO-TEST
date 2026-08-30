from search.manager import SearchManager

manager = SearchManager()

urls = manager.search("finasteride")

print(f"Trovati {len(urls)} URL\n")

for url in urls:
    print(url)