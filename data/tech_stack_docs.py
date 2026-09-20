"""
Master Tech Stack & Programming Languages Documentation Corpus for ZaidGPT.
Contains comprehensive technical Q&A, code examples, architecture patterns, and syntax
for all major languages, frameworks, AI/ML tools, cloud/DevOps, and databases.
"""

TECH_DOCS = """
User: What are the main programming languages and their primary use cases?
Assistant: The major programming languages include:
1. Python: AI/ML (PyTorch, TensorFlow), data science (Pandas), backend (FastAPI, Django), automation.
2. JavaScript / TypeScript: Frontend web (React, Next.js, Vue), backend (Node.js, Express, Bun).
3. Rust: Systems programming, high-performance engines, memory safety without garbage collection.
4. Go (Golang): Cloud-native microservices, networking, distributed systems (Kubernetes, Docker).
5. C++: Game engines (Unreal), high-frequency trading (HFT), real-time graphics, embedded systems.
6. Java & Kotlin: Enterprise backends (Spring Boot), Android development.
7. C# (.NET): Enterprise applications, game development (Unity), desktop software.
8. SQL: Relational database queries, data manipulation, and transactions.

User: Explain React and Next.js.
Assistant: React is a declarative, component-based JavaScript library for building user interfaces using Virtual DOM and JSX. Next.js is a full-stack React framework featuring:
- Server-Side Rendering (SSR) & Static Site Generation (SSG).
- App Router with React Server Components (RSC) for zero-bundle-size server execution.
- API Routes and middleware for full-stack edge computing.

User: What is TypeScript and how does it improve JavaScript?
Assistant: TypeScript is a statically typed superset of JavaScript developed by Microsoft. It compiles to plain JavaScript and adds:
- Static type checking to catch errors at compile-time before runtime execution.
- Interfaces, Generics, Enums, and Union/Intersection types.
- Enhanced IDE autocompletion and enterprise-grade refactoring support.

User: How does memory management work in Rust without a garbage collector?
Assistant: Rust enforces memory safety at compile-time through its Borrow Checker and Ownership system with three fundamental rules:
1. Each value in Rust has an owner variable.
2. There can only be one owner at a time.
3. When the owner goes out of scope, the value is automatically dropped and deallocated.
Borrowing allows references: either multiple immutable references (&T) OR exactly one mutable reference (&mut T) at any given time, preventing data races.

User: Explain PyTorch and how Tensors and Autograd work.
Assistant: PyTorch is an open-source deep learning framework based on Torch.
- Tensor: An n-dimensional array similar to NumPy ndarray, but capable of running on GPUs with CUDA acceleration.
- Autograd: Automatic differentiation engine that records a computation graph (DAG) during the forward pass. Calling loss.backward() automatically calculates gradients for all leaf tensors with requires_grad=True.

User: What is Docker and Containerization?
Assistant: Docker packages applications and all their dependencies into isolated, lightweight containers. Unlike virtual machines that emulate full guest operating systems, containers share the host OS kernel, offering instant startup times and minimal resource overhead.

User: What is Kubernetes (K8s)?
Assistant: Kubernetes is an open-source container orchestration system that automates deployment, scaling, and management of containerized applications across server clusters. Key concepts include:
- Pods: The smallest deployable computing units.
- Deployments: Declarative state managers for replica sets.
- Services: Persistent network endpoints and load balancers.
- Ingress: External routing into the cluster.

User: What is FastAPI and how does it compare to Django and Flask?
Assistant: FastAPI is a modern, high-performance Python web framework based on Starlette and Pydantic:
- Blazing fast performance comparable to NodeJS and Go using ASGI (async/await).
- Automatic data validation, serialization, and interactive Swagger/OpenAPI documentation.
- Django is a "batteries-included" monolithic framework with built-in ORM and Admin UI.
- Flask is a lightweight microframework requiring third-party libraries for ORM and auth.

User: Explain PostgreSQL vs MongoDB vs Redis.
Assistant:
1. PostgreSQL: Relational (RDBMS) database supporting ACID transactions, complex JOINs, JSONB semi-structured data, and robust indexing (B-Tree, GIN, GiST).
2. MongoDB: Document-oriented NoSQL database storing JSON-like BSON documents, ideal for rapidly evolving schemas and horizontal scaling.
3. Redis: In-memory key-value data store used for ultra-low latency caching, pub/sub messaging, session management, and rate limiting.

User: What is REST API vs GraphQL vs gRPC?
Assistant:
- REST: Uses standard HTTP methods (GET, POST, PUT, DELETE) with JSON endpoints; can suffer from over-fetching or under-fetching.
- GraphQL: Allows clients to query exact data fields in a single POST request schema; eliminates over-fetching.
- gRPC: High-performance RPC framework using HTTP/2 transport and binary Protocol Buffers (Protobuf) for ultra-fast microservice communication.

User: What is a Vector Database in AI and LLMs?
Assistant: Vector databases (e.g., Pinecone, ChromaDB, Qdrant, Milvus, pgvector) store high-dimensional mathematical embeddings generated by neural networks. They enable semantic search using Approximate Nearest Neighbor (ANN) algorithms (such as HNSW or IVF) through cosine similarity, Euclidean distance, or dot product.

User: Explain System Design: How to design a high-scale URL shortener (like Bitly).
Assistant: A scalable URL shortener system design requires:
1. API: POST /api/shorten (long_url) -> short_url; GET /{short_code} -> 301/302 Redirect.
2. Short Code Generation: Base62 encoding (a-z, A-Z, 0-9) of a unique 64-bit auto-incrementing ID or distributed Snowflake ID.
3. Database: Key-Value store or PostgreSQL with index on short_code.
4. Caching: Redis cluster caching top 20% most accessed URLs (Pareto 80/20 rule) to achieve sub-millisecond response times.
5. Load Balancer: Nginx / AWS ALB distributing traffic across stateless application instances.

User: How do JWT (JSON Web Tokens) work for authentication?
Assistant: A JWT consists of three base64url-encoded parts separated by dots: Header.Payload.Signature.
- Header: Algorithm and token type.
- Payload: User claims (user_id, expiration, role).
- Signature: Cryptographic hash HMACSHA256(header + "." + payload, secret_key) verifying the token has not been tampered with.
Stateless: The server does not need to store session state in a database to authenticate requests.

User: What is CI/CD?
Assistant: Continuous Integration (CI) automatically builds and tests code changes whenever commits are pushed. Continuous Delivery/Deployment (CD) automatically releases and deploys validated code into staging and production environments using tools like GitHub Actions, GitLab CI, or Jenkins.

User: What is an async/await event loop in JavaScript and Python?
Assistant: An event loop enables non-blocking asynchronous I/O concurrency on a single thread. When an I/O operation (network request, file read) begins, the function yields execution back to the loop, which processes other tasks until the I/O event completes, avoiding thread blocking.
"""
