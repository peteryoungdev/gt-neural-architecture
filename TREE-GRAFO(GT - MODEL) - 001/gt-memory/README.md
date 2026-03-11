# GT-Memory: Arquitetura de Memória Híbrida Grafo-Árvore

> Sistema de memória cognitiva de alto desempenho para agentes de IA autônomos.

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)

## 🧠 Visão Geral

O **GT-Memory** implementa a **Arquitetura de Memória Híbrida Grafo-Árvore (G-T)** para sistemas cognitivos de IA. Combina:

- **G-Layer**: Grafo de Conhecimento para raciocínio relacional e contexto
- **MU-Tree Layer**: Árvores hierárquicas para armazenamento factual O(log n)
- **Protocolo Ψ**: Link simbólico entre camadas com compartilhamento estrutural

```
┌─────────────────────────────────────────────────┐
│              LLMs (OpenAI, Claude, Gemini)      │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│              GT-API (REST Universal)            │
│     /memory/store  /query  /graph/traverse      │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│                  GT-Kernel                      │
│  ┌─────────┐  ┌─────────┐  ┌───────────────┐   │
│  │ G-Layer │◄─┤    Ψ    ├─►│   MU-Tree    │   │
│  │ (Graph) │  │ Protocol│  │ (Factual)    │   │
│  └─────────┘  └─────────┘  └───────────────┘   │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Pré-requisitos

- Rust 1.75+ (rustup)
- Cargo

### Instalação

```bash
# Clone o projeto
cd gt-memory

# Build
cargo build --release

# Rodar testes
cargo test --workspace

# Iniciar API
cargo run --package gt-api
```

### Uso da API

```bash
# Health check
curl http://localhost:3000/health

# Adicionar nó ao grafo
curl -X POST http://localhost:3000/graph/node \
  -H "Content-Type: application/json" \
  -d '{
    "id": "france",
    "label": "France",
    "embedding": [0.1, 0.2, 0.3, ...]
  }'

# Armazenar fato
curl -X POST http://localhost:3000/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Paris é a capital da França",
    "embedding": [0.1, 0.2, 0.3, ...],
    "graph_node_id": "france"
  }'

# Query
curl -X POST http://localhost:3000/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, 0.3, ...],
    "top_k": 5
  }'
```

## 📦 Estrutura do Projeto

```
gt-memory/
├── kernel/                 # GT-Kernel (biblioteca core)
│   └── src/
│       ├── mu_tree/        # Memory Unit Tree
│       ├── g_layer/        # Knowledge Graph
│       ├── psi/            # Protocolo Ψ
│       ├── query/          # Dual Retrieval
│       ├── sync/           # Hybrid Update
│       ├── embeddings/     # Bridge de Embeddings
│       └── storage/        # Persistência
│
└── api/                    # GT-API (REST server)
    └── src/
        ├── routes/         # Endpoints
        ├── dto/            # Request/Response DTOs
        └── middleware/     # Auth, CORS, etc
```

## 🔧 Arquitetura

### G-Layer (Knowledge Graph)

- Armazena entidades e relações em triplas RDF-like
- Multi-hop traversal para raciocínio complexo
- Embeddings combinados (relacional + Tree Encoding)

### MU-Tree (Memory Unit Tree)

- Busca semântica O(log n)
- Inserção orientada por similaridade
- Hierarquia de abstração (root = resumo, folha = fato)

### Protocolo Ψ

- Link simbólico G-Layer → MU-Tree
- Suporte a offset para sub-árvores (Ψ_offset)
- Ψ-Share para deduplicação estrutural

## 📊 Benchmarks

| Operação | Complexidade | Target |
|----------|--------------|--------|
| MU-Tree Insert | O(log n) | < 1ms |
| MU-Tree Search | O(log n) | < 100μs |
| Dual Query | O(log n + hops) | < 10ms |
| Ψ-Share Dedup | O(1) | > 40% redução |

## 🛠 API Endpoints

| Method | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/health` | Health check |
| `POST` | `/memory/store` | Armazena fato |
| `POST` | `/memory/query` | Query dual |
| `GET` | `/memory/:id` | Recupera fato |
| `POST` | `/graph/node` | Adiciona nó |
| `POST` | `/graph/edge` | Adiciona aresta |
| `POST` | `/graph/traverse` | Multi-hop traversal |
| `GET` | `/stats` | Estatísticas |

## 📄 Licença

MIT License - Ricardo Juvencio Oliveira

## 📚 Referências

- [Task Memory Engine (TME)](https://arxiv.org/abs/2504.08525)
- [MemTree: Dynamic Tree Memory Representation](https://arxiv.org/html/2410.14052v3)
- [Knowledge Graph Embeddings](https://en.wikipedia.org/wiki/Knowledge_graph_embedding)
