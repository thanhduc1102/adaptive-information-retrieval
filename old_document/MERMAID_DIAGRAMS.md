# MERMAID DIAGRAMS - System Architecture Visualization

## 1. TỔNG QUAN HỆ THỐNG (High-Level Architecture)

```mermaid
graph TB
    subgraph "Input"
        Q[Query q₀]
        GT[Ground Truth<br/>Qrels]
    end
    
    subgraph "Stage 0: Candidate Mining"
        BM25[BM25 Retrieval<br/>top-k₀ docs]
        TFIDF[TF-IDF<br/>Extractor]
        BMCS[BM25 Contribution<br/>Scorer]
        KB[KeyBERT<br/>Optional]
        C[Candidate Set C<br/>|C| ≤ 200]
        
        Q --> BM25
        BM25 --> TFIDF
        BM25 --> BMCS
        BM25 --> KB
        TFIDF --> C
        BMCS --> C
        KB --> C
    end
    
    subgraph "Stage 1: RL Query Reformulation"
        STATE[State s_t<br/>q₀, q_t, C, features]
        POLICY[Policy Network<br/>Actor-Critic]
        ACTION[Action a_t<br/>select term or STOP]
        QUERIES[m Query Variants<br/>q⁽¹⁾...q⁽ᵐ⁾]
        
        Q --> STATE
        C --> STATE
        STATE --> POLICY
        POLICY --> ACTION
        ACTION -->|repeat m times| QUERIES
    end
    
    subgraph "Stage 2: Multi-Query Retrieval + RRF"
        RET1[BM25 Retrieval<br/>L₁]
        RET2[BM25 Retrieval<br/>L₂]
        RETM[BM25 Retrieval<br/>Lₘ]
        RRF[RRF Fusion<br/>Σ 1/(k+rank)]
        TOPK[Top-K Candidates]
        
        QUERIES --> RET1
        QUERIES --> RET2
        QUERIES --> RETM
        RET1 --> RRF
        RET2 --> RRF
        RETM --> RRF
        RRF --> TOPK
    end
    
    subgraph "Stage 3: BERT Re-ranking"
        BERT[BERT Cross-Encoder<br/>MiniLM-L12]
        RERANK[Re-ranked Results]
        
        TOPK --> BERT
        Q --> BERT
        BERT --> RERANK
    end
    
    subgraph "Evaluation"
        METRICS[Metrics<br/>MRR@10, Recall@100<br/>nDCG@10, Latency]
        
        RERANK --> METRICS
        GT --> METRICS
    end
    
    style Q fill:#e1f5ff
    style RERANK fill:#c8e6c9
    style METRICS fill:#fff9c4
```

## 2. RL TRAINING LOOP

```mermaid
sequenceDiagram
    participant Env as Environment
    participant Agent as RL Agent
    participant Search as Search Engine
    participant Buffer as Replay Buffer
    
    loop Each Episode
        Env->>Agent: state s₀ = (q₀, C, features)
        
        loop t = 1 to T (max steps)
            Agent->>Agent: π_θ(a_t | s_t)
            Agent->>Env: action a_t (term or STOP)
            
            alt a_t == STOP
                Env->>Search: query q_final
                Search->>Env: results + metrics
                Env->>Agent: reward R_t
                Note over Agent: Episode ends
            else a_t == term
                Env->>Env: q_t+1 = q_t ⊕ a_t
                Env->>Agent: s_t+1, r_t = 0
            end
        end
        
        Agent->>Buffer: store (s, a, r, s')
        
        alt Buffer full
            Buffer->>Agent: sample minibatch
            Agent->>Agent: PPO update<br/>L = L_policy + L_value + L_entropy
        end
    end
```

## 3. DATA FLOW PIPELINE

```mermaid
flowchart LR
    subgraph "Data Sources"
        MSM[(MS MARCO<br/>Passages)]
        JEP[(Jeopardy<br/>HDF5)]
        TRC[(TREC-CAR<br/>HDF5)]
        WV[(Word2Vec<br/>374K)]
    end
    
    subgraph "Preprocessing"
        IDX[Lucene/Pyserini<br/>Indexing]
        TOK[Tokenization<br/>NLTK]
        EMB[Embedding<br/>Lookup]
    end
    
    subgraph "Training Data"
        QTR[Queries<br/>Train/Valid/Test]
        DID[Doc IDs<br/>Ground Truth]
        FEAT[Features<br/>IDF, TF, BM25]
    end
    
    subgraph "Model Inputs"
        BATCH[Batched<br/>Episodes]
    end
    
    MSM --> IDX
    JEP --> IDX
    TRC --> IDX
    
    MSM --> TOK
    JEP --> TOK
    TRC --> TOK
    
    WV --> EMB
    TOK --> EMB
    
    TOK --> QTR
    TOK --> DID
    IDX --> FEAT
    
    QTR --> BATCH
    DID --> BATCH
    FEAT --> BATCH
    EMB --> BATCH
    
    style BATCH fill:#ffccbc
```

## 4. RL AGENT ARCHITECTURE

```mermaid
graph TB
    subgraph "Input Layer"
        Q0[Query q₀<br/>Embedding<br/>512-dim]
        QT[Query q_t<br/>Embedding<br/>512-dim]
        CAND[Candidates C<br/>N × 128 features]
    end
    
    subgraph "Encoder"
        CONCAT[Concatenate<br/>[q₀ ∥ q_t ∥ C]]
        TRANS[Transformer Encoder<br/>d=256, heads=4<br/>layers=2]
        ATT[Attention<br/>Q=q_t, K=C, V=C]
    end
    
    subgraph "Actor Head"
        ACT_FC1[Linear 256→128]
        ACT_RELU[ReLU]
        ACT_FC2[Linear 128→|C|+1]
        ACT_SOFT[Softmax]
        ACTION[Action Probs<br/>π_θ(a|s)]
    end
    
    subgraph "Critic Head"
        CRIT_FC1[Linear 256→128]
        CRIT_RELU[ReLU]
        CRIT_FC2[Linear 128→1]
        VALUE[Value V_ϕ(s)]
    end
    
    Q0 --> CONCAT
    QT --> CONCAT
    CAND --> CONCAT
    CONCAT --> TRANS
    TRANS --> ATT
    
    ATT --> ACT_FC1
    ACT_FC1 --> ACT_RELU
    ACT_RELU --> ACT_FC2
    ACT_FC2 --> ACT_SOFT
    ACT_SOFT --> ACTION
    
    ATT --> CRIT_FC1
    CRIT_FC1 --> CRIT_RELU
    CRIT_RELU --> CRIT_FC2
    CRIT_FC2 --> VALUE
    
    style ACTION fill:#81c784
    style VALUE fill:#64b5f6
```

## 5. RRF FUSION ALGORITHM

```mermaid
flowchart TD
    START([Start]) --> INPUT[Input: m ranked lists<br/>L₁, L₂, ..., Lₘ]
    INPUT --> INIT[Initialize:<br/>doc_scores = {}]
    INIT --> LOOP1{For each<br/>list i}
    
    LOOP1 -->|i ≤ m| LOOP2{For each<br/>doc d in Lᵢ}
    LOOP2 -->|has next| GET[Get rank_i(d)]
    GET --> CALC[score += 1/(k + rank_i)]
    CALC --> UPDATE[doc_scores[d] += score]
    UPDATE --> LOOP2
    
    LOOP2 -->|done| LOOP1
    LOOP1 -->|done| SORT[Sort by score DESC]
    SORT --> OUTPUT[Return merged list]
    OUTPUT --> END([End])
    
    style START fill:#e1bee7
    style END fill:#c5e1a5
```

## 6. EVALUATION PIPELINE

```mermaid
graph LR
    subgraph "Test Data"
        Q_TEST[Test Queries]
        QRELS[Relevance<br/>Judgments]
    end
    
    subgraph "System Under Test"
        SYS[Retrieve-Fuse<br/>-Re-rank System]
    end
    
    subgraph "Metrics Computation"
        REC[Recall@100<br/>Calculator]
        MRR[MRR@10<br/>Calculator]
        NDCG[nDCG@10<br/>Calculator]
        LAT[Latency<br/>Profiler]
    end
    
    subgraph "Results"
        REP[Evaluation Report<br/>Tables + Plots]
    end
    
    Q_TEST --> SYS
    SYS --> REC
    SYS --> MRR
    SYS --> NDCG
    SYS --> LAT
    
    QRELS --> REC
    QRELS --> MRR
    QRELS --> NDCG
    
    REC --> REP
    MRR --> REP
    NDCG --> REP
    LAT --> REP
    
    style REP fill:#fff59d
```

## 7. TRAINING vs INFERENCE MODE

```mermaid
stateDiagram-v2
    [*] --> Training
    [*] --> Inference
    
    state Training {
        [*] --> SampleAction
        SampleAction --> ExecuteAction
        ExecuteAction --> ComputeReward
        ComputeReward --> UpdatePolicy
        UpdatePolicy --> SampleAction
        
        note right of UpdatePolicy
            PPO update:
            - Clip ratio
            - Value loss
            - Entropy bonus
        end note
    }
    
    state Inference {
        [*] --> GreedyAction
        GreedyAction --> ExecuteAction_Inf
        ExecuteAction_Inf --> NextQuery
        NextQuery --> GreedyAction
        
        note right of GreedyAction
            Greedy selection:
            a* = argmax π_θ(a|s)
        end note
    }
    
    Training --> [*]: Converged
    Inference --> [*]: All queries done
```

## 8. ABLATION STUDY STRUCTURE

```mermaid
graph TD
    FULL[Full System<br/>RL + Multi-Query + RRF + BERT]
    
    ABL1[Ablation 1<br/>No RL → Random term selection]
    ABL2[Ablation 2<br/>No RRF → Single best query]
    ABL3[Ablation 3<br/>No BERT → BM25 only]
    ABL4[Ablation 4<br/>m=1,2,4,8 variants]
    ABL5[Ablation 5<br/>Different rewards]
    
    BASE1[Baseline: BM25]
    BASE2[Baseline: BM25 + RM3]
    BASE3[Baseline: BM25 → BERT]
    
    FULL --> COMPARE{Compare<br/>Metrics}
    ABL1 --> COMPARE
    ABL2 --> COMPARE
    ABL3 --> COMPARE
    ABL4 --> COMPARE
    ABL5 --> COMPARE
    BASE1 --> COMPARE
    BASE2 --> COMPARE
    BASE3 --> COMPARE
    
    COMPARE --> INSIGHT[Insights:<br/>- Component contribution<br/>- Best configuration<br/>- Bottlenecks]
    
    style FULL fill:#4caf50
    style COMPARE fill:#ff9800
    style INSIGHT fill:#2196f3
```

## 9. ERROR HANDLING & RECOVERY

```mermaid
flowchart TD
    START([Query Input]) --> VAL{Valid<br/>Query?}
    
    VAL -->|Yes| MINE[Candidate Mining]
    VAL -->|No| ERR1[Error: Empty query]
    
    MINE --> CHECK1{Candidates<br/>found?}
    CHECK1 -->|Yes| RL[RL Reformulation]
    CHECK1 -->|No| FALLBACK1[Fallback: Use q₀]
    
    RL --> CHECK2{Valid<br/>reformulation?}
    CHECK2 -->|Yes| RETR[Multi-Query Retrieval]
    CHECK2 -->|No| FALLBACK2[Fallback: Single query]
    
    RETR --> CHECK3{Results<br/>found?}
    CHECK3 -->|Yes| RRF[RRF Fusion]
    CHECK3 -->|No| ERR2[Error: No results]
    
    RRF --> BERT[BERT Re-rank]
    BERT --> OUT([Final Results])
    
    FALLBACK1 --> RETR
    FALLBACK2 --> RETR
    ERR1 --> LOG[Log Error]
    ERR2 --> LOG
    LOG --> OUT
    
    style ERR1 fill:#f44336
    style ERR2 fill:#f44336
    style OUT fill:#4caf50
```

## 10. DEPLOYMENT ARCHITECTURE

```mermaid
C4Context
    title Deployment Architecture - Production System
    
    Person(user, "User", "Submits search query")
    
    System_Boundary(sys, "IR System") {
        Container(api, "API Gateway", "FastAPI", "REST endpoints")
        Container(rl, "RL Service", "PyTorch", "Query reformulation")
        Container(search, "Search Service", "Pyserini", "BM25 retrieval")
        Container(bert, "Re-rank Service", "Transformers", "BERT scoring")
        ContainerDb(cache, "Redis Cache", "Cache", "Results cache")
        ContainerDb(index, "Lucene Index", "Index", "Document index")
    }
    
    System_Ext(monitor, "Monitoring", "Prometheus + Grafana")
    
    Rel(user, api, "HTTP POST /search")
    Rel(api, cache, "Check cache")
    Rel(api, rl, "Get query variants")
    Rel(rl, search, "Multi-query retrieval")
    Rel(search, index, "Query index")
    Rel(api, bert, "Re-rank top-K")
    Rel(bert, api, "Ranked results")
    Rel(api, user, "JSON response")
    Rel(api, monitor, "Metrics")
    
    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## USAGE INSTRUCTIONS

### Viewing Mermaid Diagrams

1. **VS Code**: Install "Markdown Preview Mermaid Support" extension
2. **GitHub**: Diagrams render automatically in README.md
3. **Online**: Copy to https://mermaid.live/

### Customizing Diagrams

- Edit node colors: `style NodeID fill:#colorcode`
- Change flow direction: `TB` (top-bottom), `LR` (left-right)
- Add notes: `note right of Node: text`
