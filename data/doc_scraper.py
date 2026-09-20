"""
Universal Documentation & Coding Scraper Service for ZaidGPT.
Downloads full programming documentation, cheatsheets, data structures,
algorithms, and architectural blueprints directly in plain text format.
"""

import argparse
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List


DOCS_CATALOG: Dict[str, List[Dict[str, str]]] = {
    "algorithms_and_structures": [
        {
            "name": "Singly Linked List",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/linked_list/singly_linked_list.py",
        },
        {
            "name": "Binary Search Tree",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/binary_search_tree.py",
        },
        {
            "name": "QuickSort",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/quick_sort.py",
        },
        {
            "name": "MergeSort",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/merge_sort.py",
        },
        {
            "name": "Dijkstra Graph Algorithm",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/dijkstra.py",
        },
        {
            "name": "Breadth First Search (BFS)",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/graphs/breadth_first_search.py",
        },
        {
            "name": "0/1 Knapsack Dynamic Programming",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/dynamic_programming/knapsack.py",
        },
        {
            "name": "RSA Cryptography",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/ciphers/rsa_cipher.py",
        },
        {
            "name": "Matrix Multiplication",
            "url": "https://raw.githubusercontent.com/TheAlgorithms/Python/master/maths/matrix_multiplication.py",
        },
    ]
}


def clean_markdown_text(raw_text: str, name: str) -> str:
    cleaned = re.sub(r"^---[\s\S]*?---", "", raw_text)
    cleaned = cleaned.strip()

    formatted = f"""
# ==============================================================================
# DOCUMENTATION & SOURCE CODE MODULE: {name.upper()}
# ==============================================================================
{cleaned}
"""
    return formatted.strip()


class DocScraperService:
    def __init__(self, output_dir: str = "data/raw_docs", target_train_file: str = "data/train.txt"):
        self.output_dir = Path(output_dir)
        self.target_train_file = Path(target_train_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_train_file.parent.mkdir(parents=True, exist_ok=True)

    def fetch_url(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ZaidGPT-DocBot"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode("utf-8", errors="replace")

    def scrape_category(self, category: str) -> List[str]:
        if category not in DOCS_CATALOG:
            print(f"[!] Unknown category '{category}'. Available: {list(DOCS_CATALOG.keys())}", flush=True)
            return []

        docs = DOCS_CATALOG[category]
        collected = []

        print(f"\n[*] Scraping category: {category.upper()} ({len(docs)} sources)...", flush=True)
        for item in docs:
            name = item["name"]
            url = item["url"]
            try:
                raw_text = self.fetch_url(url)
                cleaned = clean_markdown_text(raw_text, name)

                safe_name = re.sub(r"[^\w\-_\. ]", "_", name.lower())
                doc_path = self.output_dir / f"{safe_name}.txt"
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                collected.append(cleaned)
                print(f"  [+] Fetched {name} ({len(cleaned):,} characters) -> {doc_path}", flush=True)
                time.sleep(0.1)
            except Exception as e:
                print(f"  [-] Failed to fetch {name}: {e}", flush=True)

        return collected

    def scrape_all(self, append_to_training: bool = True):
        all_docs = []
        for category in DOCS_CATALOG:
            category_docs = self.scrape_category(category)
            all_docs.extend(category_docs)

        if append_to_training and all_docs:
            combined_corpus = "\n\n" + "\n\n".join(all_docs) + "\n"
            with open(self.target_train_file, "a", encoding="utf-8") as f:
                f.write(combined_corpus)
            print("\n" + "=" * 60, flush=True)
            print(f"[SUCCESS] Scraped {len(all_docs)} full documentation & source modules!", flush=True)
            print(f"[SUCCESS] Injected {len(combined_corpus):,} characters into {self.target_train_file}", flush=True)
            print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZaidGPT Coding Documentation Scraper")
    parser.add_argument("--all", action="store_true", help="Download all documentation in the catalog")
    parser.add_argument("--no_append", action="store_true", help="Do not append to data/train.txt")
    args = parser.parse_args()

    scraper = DocScraperService()
    scraper.scrape_all(append_to_training=(not args.no_append))
