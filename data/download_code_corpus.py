"""
Automated Web Coding Resources & Architecture Ingestion Engine for ZaidGPT.
Downloads and compiles real-world source code, Design Patterns, Algorithms,
System Architecture, and Web Framework templates from open educational repositories.
"""

import os
import urllib.request
from pathlib import Path

# Deep coding patterns, software architecture, and algorithm blueprints
CODE_AND_ARCHITECTURE_BLUEPRINTS = """
User: Write a complete LRU (Least Recently Used) Cache implementation in Python.
Assistant: An LRU Cache uses a Hash Map combined with a Doubly Linked List to achieve O(1) time complexity for both get() and put() operations. Here is the complete implementation:

class Node:
    def __init__(self, key: int, value: int):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node
        self.head = Node(0, 0)  # Dummy head (Most Recent)
        self.tail = Node(0, 0)  # Dummy tail (Least Recent)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node: Node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add_to_front(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add_to_front(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

User: How do you build a production-grade REST API with FastAPI, Pydantic, and SQLite?
Assistant: Here is a complete, runnable FastAPI CRUD service with request validation and database models:

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="ZaidGPT API", version="1.0.0")

class ItemSchema(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    price: float = Field(..., gt=0.0)

class ItemResponse(ItemSchema):
    id: int

fake_db = {}
id_counter = 1

@app.post("/items/", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemSchema):
    global id_counter
    new_item = ItemResponse(id=id_counter, **item.dict())
    fake_db[id_counter] = new_item
    id_counter += 1
    return new_item

@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int):
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return fake_db[item_id]

@app.get("/items/", response_model=List[ItemResponse])
async def list_items():
    return list(fake_db.values())

User: Explain the Factory and Observer Design Patterns with code examples.
Assistant:
1. Factory Pattern: Creational design pattern that provides an interface for creating objects in a superclass, allowing subclasses to alter the type of objects created.

class Dog:
    def speak(self): return "Woof!"

class Cat:
    def speak(self): return "Meow!"

class PetFactory:
    @staticmethod
    def get_pet(pet_type: str):
        pets = {"dog": Dog(), "cat": Cat()}
        return pets.get(pet_type.lower())

2. Observer Pattern: Behavioral design pattern defining a subscription mechanism to notify multiple objects about any events that happen to the object they are observing.

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def notify(self, message: str):
        for observer in self._observers:
            observer.update(message)

class UserObserver:
    def __init__(self, name: str):
        self.name = name

    def update(self, message: str):
        print(f"[{self.name}] Received notification: {message}")

User: Write a complete Dijkstra's Shortest Path Algorithm in Python.
Assistant: Dijkstra's algorithm finds the shortest path between nodes in a weighted graph with non-negative edge weights using a Min-Heap (Priority Queue):

import heapq

def dijkstra(graph, start_node):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    pq = [(0, start_node)]  # (distance, node)

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

User: What is Event-Driven Microservices Architecture?
Assistant: In an Event-Driven Architecture (EDA), decoupled services communicate asynchronously by producing and consuming events via message brokers like Apache Kafka, RabbitMQ, or AWS SQS/SNS.
Key benefits:
1. Loose Coupling: Services operate independently without direct point-to-point HTTP dependencies.
2. Fault Tolerance: If a consumer service goes down, messages buffer in the queue until recovery without losing transactions.
3. Scalability: Producers and consumers scale independently based on throughput demand.
"""


def fetch_online_code_corpus():
    """
    Downloads open-source Python code algorithms and documentation datasets from GitHub.
    """
    print("[Code Ingestion] Fetching open-source algorithms and programming documentation...")
    sources = [
        # The Algorithms Python repository sample (Clean, well-documented Python algorithms)
        "https://raw.githubusercontent.com/TheAlgorithms/Python/master/sorts/quick_sort.py",
        "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/binary_search.py",
        "https://raw.githubusercontent.com/TheAlgorithms/Python/master/data_structures/binary_tree/binary_search_tree.py",
    ]

    downloaded_code = []
    for url in sources:
        try:
            filename = url.split("/")[-1]
            req = urllib.request.Request(url, headers={"User-Agent": "ZaidGPT-CodeBot"})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode("utf-8")
                downloaded_code.append(f"\n# Code Module: {filename}\n" + content)
                print(f"[*] Successfully fetched: {filename}", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not fetch {url}: {e}", flush=True)

    return "\n\n".join(downloaded_code)


def assemble_all_code_and_architecture():
    print("[Code Ingestion] Compiling Master Code & Architecture Corpus...")
    online_code = fetch_online_code_corpus()
    full_corpus = CODE_AND_ARCHITECTURE_BLUEPRINTS + "\n\n" + online_code

    # Append to data/train.txt
    data_file = Path("data/train.txt")
    with open(data_file, "a", encoding="utf-8") as f:
        f.write("\n\n" + full_corpus.strip() + "\n")

    print(f"[Code Ingestion] Successfully injected {len(full_corpus):,} characters of coding blueprints into data/train.txt!")


if __name__ == "__main__":
    assemble_all_code_and_architecture()
