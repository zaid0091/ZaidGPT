"""
Real Official Documentation Scraper for ZaidGPT.
Downloads full, uncompressed, authentic official documentation directly from
the source repositories of Python, Rust, FastAPI, PyTorch, React, and System Design.
"""

import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List


OFFICIAL_DOCS_SOURCES: Dict[str, List[Dict[str, str]]] = {
    "system_design": [
        {
            "title": "System Design Primer Core Architecture",
            "url": "https://raw.githubusercontent.com/donnemartin/system-design-primer/master/README.md",
        },
    ],
    "rust_official_book": [
        {
            "title": "Rust Ownership & Borrowing",
            "url": "https://raw.githubusercontent.com/rust-lang/book/main/src/ch04-01-what-is-ownership.md",
        },
        {
            "title": "Rust References and Borrowing",
            "url": "https://raw.githubusercontent.com/rust-lang/book/main/src/ch04-02-references-and-borrowing.md",
        },
        {
            "title": "Rust Structs and Data",
            "url": "https://raw.githubusercontent.com/rust-lang/book/main/src/ch05-01-defining-structs.md",
        },
        {
            "title": "Rust Enums and Pattern Matching",
            "url": "https://raw.githubusercontent.com/rust-lang/book/main/src/ch06-01-defining-an-enum.md",
        },
        {
            "title": "Rust Error Handling",
            "url": "https://raw.githubusercontent.com/rust-lang/book/main/src/ch09-02-recoverable-errors-with-result.md",
        },
        {
            "title": "Rust Fearless Concurrency",
            "url": "https://raw.githubusercontent.com/rust-lang/book/main/src/ch16-01-threads.md",
        },
    ],
    "fastapi_official_docs": [
        {
            "title": "FastAPI First Steps",
            "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/first-steps.md",
        },
        {
            "title": "FastAPI Path Parameters",
            "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/path-params.md",
        },
        {
            "title": "FastAPI Query Parameters",
            "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/query-params.md",
        },
        {
            "title": "FastAPI Request Body and Fields",
            "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/body-fields.md",
        },
        {
            "title": "FastAPI Dependencies Injection",
            "url": "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs/tutorial/dependencies/index.md",
        },
    ],
    "pytorch_official_docs": [
        {
            "title": "PyTorch Tensor Deep Dive",
            "url": "https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/basics/tensorqs_tutorial.py",
        },
        {
            "title": "PyTorch Autograd Mechanics",
            "url": "https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/basics/autogradqs_tutorial.py",
        },
        {
            "title": "PyTorch Neural Networks Architecture",
            "url": "https://raw.githubusercontent.com/pytorch/tutorials/main/beginner_source/basics/buildmodel_tutorial.py",
        },
    ],
}


class RealWebDocsScraper:
    def __init__(self, output_dir: str = "data/raw_docs/official", target_train_file: str = "data/train.txt"):
        self.output_dir = Path(output_dir)
        self.target_train_file = Path(target_train_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, url: str) -> str:
        headers = {"User-Agent": "ZaidGPT-OfficialDocsScraper/2.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def run(self):
        print("\n" + "=" * 60)
        print("[*] SCRAPING AUTHENTIC OFFICIAL DOCUMENTATION REPOSITORIES")
        print("=" * 60 + "\n", flush=True)

        all_content = []
        total_chars = 0

        for section_name, sources in OFFICIAL_DOCS_SOURCES.items():
            print(f"[*] Fetching Category: {section_name.upper()} ({len(sources)} modules)...")
            for item in sources:
                title = item["title"]
                url = item["url"]
                try:
                    raw_text = self.fetch(url)
                    # Clean markdown tags
                    clean_text = re.sub(r"<!--[\s\S]*?-->", "", raw_text).strip()
                    
                    header = f"\n\n# ==============================================================================\n# OFFICIAL DOC: {title.upper()}\n# SOURCE: {url}\n# ==============================================================================\n"
                    formatted_doc = header + clean_text
                    
                    # Save local copy
                    safe_filename = re.sub(r"[^\w\-_\.]", "_", title.lower()) + ".md"
                    file_path = self.output_dir / safe_filename
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(formatted_doc)

                    all_content.append(formatted_doc)
                    total_chars += len(formatted_doc)
                    print(f"  [+] Saved: {title} ({len(formatted_doc):,} chars) -> {file_path}")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  [-] Failed to fetch {title}: {e}")

        # Append to target training dataset
        if all_content:
            master_append = "\n\n".join(all_content)
            with open(self.target_train_file, "a", encoding="utf-8") as f:
                f.write(master_append)

            print("\n" + "=" * 60)
            print(f"[SUCCESS] Official Documentation Scrape Complete!")
            print(f"[SUCCESS] Total Ingested: {total_chars:,} characters")
            print(f"[SUCCESS] Master training file updated: {self.target_train_file}")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    scraper = RealWebDocsScraper()
    scraper.run()
