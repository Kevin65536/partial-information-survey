# Rethinking the Academic Survey Paradigm

## Introduction
Traditional academic surveys often suffer from being static lists of summaries, lacking deep synthesis and dynamic updates. This document proposes a new paradigm for conducting and presenting literature surveys, specifically tailored for complex, evolving fields like Partial Information Decomposition (PID).

## Core Principles

### 1. Question-Driven Synthesis
Instead of organizing by paper or chronological order, the survey should be organized by **core research questions**.
- **What is the fundamental problem?** (e.g., How to quantify redundancy?)
- **What are the competing hypotheses/approaches?** (e.g., Optimization vs. Algebraic vs. Geometric)
- **What is the evidence?** (Key results from papers supporting each approach)

### 2. Dynamic Knowledge Graphs
Move beyond linear text. Use knowledge graphs (like `markmap`) to visualize relationships:
- **Nodes**: Concepts, Methods, Papers, Authors.
- **Edges**: "Proposes", "Refutes", "Extends", "Applies to".
- **Benefit**: Allows seeing the "genealogy" of ideas and identifying isolated clusters (gaps).

### 3. Living Documents
The survey should not be a one-time snapshot. It should be a "living document" that evolves.
- **Continuous Integration**: New papers are "committed" to the knowledge base.
- **Versioned Insights**: Tracking how the consensus on a topic changes over time.

### 4. Multi-Level Granularity
- **Level 1: Executive Summary**: High-level trends and key takeaways for quick consumption.
- **Level 2: Thematic Deep Dives**: Detailed analysis of specific sub-fields (e.g., PID in Neuroscience).
- **Level 3: Technical Specifications**: Algorithmic details, code implementations, and mathematical proofs.

## Implementation Strategy for PID Survey

1.  **Refactor Existing Notes**: Convert current "paper-by-paper" notes into "topic-based" notes.
2.  **Build the Graph**: Update the `markmap` visualizations to reflect the new structure.
3.  **Identify "Dark Matter"**: Explicitly highlight what is *unknown* or *unaddressed* in the current literature (the gaps).
4.  **Active Decoding**: Instead of just reading, actively "decode" papers by implementing their methods on toy datasets to verify claims.

## Conclusion
By shifting from a "collection of summaries" to a "synthesis of knowledge," we can create a survey that is not just a reference, but a tool for generating new research insights.
