# SciFact Dataset Provenance

This document provides provenance and metadata for the dataset used in the ResearchGuard AI Phase 5 Controlled Experiment.

## Dataset Overview

- **Dataset Name**: SciFact
- **Purpose**: Evaluates the veracity of scientific claims against scientific abstracts.
- **Source/Provenance**: Official AllenAI SciFact repository.
- **Licensing/Provenance Information**: Standard SciFact licensing (typically CC-BY or similar Open Access terms as designated by AllenAI). *The dataset itself is excluded from Git to comply with distribution best practices and maintain repository hygiene.*

## Corpus and Evidence
- **Corpus Size**: 437 unique scientific abstracts (documents) utilized in our binary mapping.
- **Unique Claims**: 693 unique claims.
- **Evidence-Pair Count**: 1,295 labeled claim-evidence pairs.

## Class Distribution
The extracted binary dataset has a natural class imbalance:
- **SUPPORT (1)**: 832 instances (64.25%)
- **CONTRADICT (0)**: 463 instances (35.75%)

*(Note: "NOT ENOUGH INFO" and other non-binary labels were excluded from this binary evaluation.)*

## Claim/Document Relationships and Leakage Prevention
Because multiple claims can be derived from the same document, and multiple documents can provide evidence for the same claim, the dataset forms a bipartite graph.

**Connected-Component Grouping**:
To prevent data leakage during cross-validation, we utilize a Disjoint Set Union (Union-Find) algorithm to map the bipartite graph into **388 independent connected components**.

**Leakage Prevention Methodology**:
During our 5-fold Stratified Group K-Fold cross-validation, these connected components act as grouping IDs. We explicitly assert programmatically that:
- Zero group leakage occurs.
- Zero Document ID leakage occurs.
- Zero Claim ID leakage occurs.
- Zero Claim text leakage occurs.
- Zero Rationale text leakage occurs.

## Clarification: SciFact vs. Runtime
It is critical to distinguish the **SciFact Controlled Evaluation** from the **ResearchGuard Runtime**:

- **SciFact Evaluation**: Evaluates short, dense scientific claims against curated scientific abstracts with known ground-truth sentences.
- **ResearchGuard Runtime**: Evaluates user-generated claims against arbitrary PDF chunks generated via local RAG (Retrieval-Augmented Generation). 

Metrics derived from the SciFact dataset represent performance on the controlled experimental task and do *not* directly represent the accuracy of arbitrary PDF verification in the runtime application.
