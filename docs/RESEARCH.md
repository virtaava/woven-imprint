# Research Foundations

Woven Imprint's architecture is informed by 471 academic papers across 6 research domains.
Key papers that shaped the design:

- **Park et al. (2023)** — [Generative Agents: Interactive Simulacra of Human Behavior](https://doi.org/10.1145/3586183.3606763)
  Memory stream architecture, retrieval function (recency x importance x relevance), reflection mechanism. The foundation for our three-tier memory model.

- **Chheda et al. (2025)** — [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://doi.org/10.3233/faia251160)
  Dual vector + graph memory storage, memory extraction pipeline, conflict resolution. Informed our hybrid retrieval and belief revision system.

- **Cartisien (2026)** — [Engram: A Local-First Persistent Memory Architecture](https://doi.org/10.5281/zenodo.18988892)
  Three-tier memory lifecycle, multi-strategy retrieval via Reciprocal Rank Fusion, belief-revision with certainty scores. Direct influence on our RRF retrieval and local-first SQLite approach.

- **Kwon et al. (2024)** — ["My agent understands me better": Dynamic Human-like Memory Recall and Consolidation](https://doi.org/10.1145/3613905.3650839)
  Human-like memory architecture with sensory/short-term/long-term tiers and autonomous recall.

- **Huang et al. (2025)** — [Post Persona Alignment for Multi-Session Dialogue Generation](https://arxiv.org/abs/2506.11857)
  Response-as-query persona retrieval, two-stage generation with post-hoc alignment. Shaped our consistency checking approach.

- **Song et al. (2020)** — [Generate, Delete and Rewrite: Persona Consistency in Dialogue](https://arxiv.org/abs/2004.07672)
  Three-stage framework for persona-consistent dialogue generation.

- **Welleck et al. (2019)** — [Generating Persona Consistent Dialogues by Exploiting NLI](https://arxiv.org/abs/1911.05889)
  NLI as RL reward for persona consistency. Inspired our NLI-based consistency checker.

- **Li et al. (2025)** — [MoCoRP: Modeling Consistent Relations between Persona and Response](https://arxiv.org/abs/2512.07544)
  Explicit NLI relation extraction between persona and response.

Full research synthesis: [research/synthesis.md](../research/synthesis.md)
