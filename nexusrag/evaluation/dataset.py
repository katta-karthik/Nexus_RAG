import json
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EvalSample:
    id: str
    question: str
    document: str
    expected_page: int
    reference_keywords: List[str]
    ground_truth_answer: str


BENCHMARK_DATASET: List[Dict[str, Any]] = [
    {
        "id": "eval_01",
        "question": "What is the primary architectural difference between the Transformer and previous sequence transduction models?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["self-attention", "recurrent", "convolution", "parallelization", "transduction"],
        "ground_truth_answer": "The Transformer relies entirely on self-attention to compute representations without using sequence-aligned RNNs or convolution, allowing significantly more parallelization.",
    },
    {
        "id": "eval_02",
        "question": "How many layers are in the Transformer encoder and what are the sub-layers?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["stack of N = 6", "multi-head self-attention", "feed-forward", "residual connection", "layer normalization"],
        "ground_truth_answer": "The encoder consists of N = 6 identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network.",
    },
    {
        "id": "eval_03",
        "question": "Why is the dot product scaled by the square root of d_k in scaled dot-product attention?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["scaling factor", "sqrt(d_k)", "magnitude", "small gradients", "softmax function"],
        "ground_truth_answer": "For large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients.",
    },
    {
        "id": "eval_04",
        "question": "How many attention heads are used and what is the dimensionality of each head?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["h = 8 heads", "d_k = d_v = 64", "d_model = 512"],
        "ground_truth_answer": "The model uses h = 8 attention heads, where d_k = d_v = d_model / h = 64.",
    },
    {
        "id": "eval_05",
        "question": "Why are positional encodings added to the input embeddings?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["positional encodings", "recurrence", "convolution", "order of the sequence", "sinusoidal"],
        "ground_truth_answer": "Because the model contains no recurrence and no convolution, positional encodings must be injected so the model can utilize sequence order.",
    },
    {
        "id": "eval_06",
        "question": "What optimizer and learning rate schedule were used to train the Transformer?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["Adam optimizer", "warmup_steps = 4000", "inverse square root", "beta_1 = 0.9"],
        "ground_truth_answer": "The Adam optimizer was used with warmup for 4000 steps, decreasing proportionally to the inverse square root of the step number thereafter.",
    },
    {
        "id": "eval_07",
        "question": "What BLEU score did the big Transformer achieve on WMT 2014 English-to-German?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["BLEU score of 28.4", "English-to-German", "P100 GPUs", "3.5 days"],
        "ground_truth_answer": "The Transformer (big) achieved a BLEU score of 28.4 on WMT 2014 English-to-German translation.",
    },
    {
        "id": "eval_08",
        "question": "What BLEU score was achieved on the WMT 2014 English-to-French translation task?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["BLEU score of 41.0", "English-to-French", "1/4 the training cost"],
        "ground_truth_answer": "The model achieved a BLEU score of 41.0 on WMT 2014 English-to-French translation.",
    },
    {
        "id": "eval_09",
        "question": "What is the computational complexity per layer of self-attention?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["O(n^2 * d)", "sequence length", "computational complexity", "restricted self-attention"],
        "ground_truth_answer": "The computational complexity per layer is O(n^2 * d), where n is the sequence length.",
    },
    {
        "id": "eval_10",
        "question": "What dropout rate and label smoothing values were applied during training?",
        "document": "attention_is_all_you_need_summary.txt",
        "expected_page": 1,
        "reference_keywords": ["P_drop = 0.1", "epsilon_ls = 0.1", "label smoothing", "perplexity"],
        "ground_truth_answer": "Residual dropout of P_drop = 0.1 and label smoothing of epsilon_ls = 0.1 were applied.",
    },
    {
        "id": "eval_11",
        "question": "What was TechCorp's total annual revenue and growth rate in Fiscal Year 2025?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["$14.8 billion", "24.5%", "FY2024", "$11.9 billion"],
        "ground_truth_answer": "TechCorp's total revenue reached $14.8 billion in FY2025, up 24.5% year-over-year.",
    },
    {
        "id": "eval_12",
        "question": "What was TechCorp's operating margin and free cash flow in FY2025?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["operating margin of 24.5%", "$3.62 billion", "Free cash flow surged by 31%", "$2.85 billion"],
        "ground_truth_answer": "TechCorp achieved an operating margin of 24.5% and free cash flow of $2.85 billion.",
    },
    {
        "id": "eval_13",
        "question": "How much revenue did the Cloud and AI Infrastructure segment generate?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["$7.2 billion", "38% YoY", "Nexus Cloud", "hyperscale compute"],
        "ground_truth_answer": "Cloud & AI Infrastructure generated $7.2 billion in revenue, representing 38% YoY growth.",
    },
    {
        "id": "eval_14",
        "question": "What was the renewal rate and Net Revenue Retention for Enterprise Software Platforms?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["renewal rates", "96.2%", "Net Revenue Retention", "118%", "Fortune 500"],
        "ground_truth_answer": "Renewal rate was 96.2% and Net Revenue Retention (NRR) was 118% among Fortune 500 customers.",
    },
    {
        "id": "eval_15",
        "question": "How much capital did TechCorp invest into Research and Development in FY2025?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["$2.6 billion", "17.6% of gross revenue", "Research & Development"],
        "ground_truth_answer": "TechCorp invested $2.6 billion into R&D, accounting for 17.6% of gross revenue.",
    },
    {
        "id": "eval_16",
        "question": "What are the major semiconductor and hardware risks facing TechCorp?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["semiconductor allocation", "HBM3e", "high-bandwidth memory", "packaging lithography", "server clusters"],
        "ground_truth_answer": "Tight supply of high-bandwidth memory (HBM3e) and advanced packaging lithography could delay delivery of server clusters in Q1/Q2 FY2026.",
    },
    {
        "id": "eval_17",
        "question": "What cybersecurity threat vector experienced a 42% surge?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["prompt injection", "privilege escalation", "surged by 42%", "enterprise agents"],
        "ground_truth_answer": "Threat vectors targeting prompt injection and privilege escalation surged by 42% as customers deployed enterprise agents.",
    },
    {
        "id": "eval_18",
        "question": "What were TechCorp's cash reserves and long-term debt at year end 2025?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["$5.4 billion in cash", "$2.1 billion", "debt", "interest rate of 3.8%"],
        "ground_truth_answer": "TechCorp held $5.4 billion in cash and marketable securities, with long-term debt of $2.1 billion.",
    },
    {
        "id": "eval_19",
        "question": "What share repurchase program and quarterly dividend did the Board authorize?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["$2.0 billion share repurchase", "quarterly cash dividend of $0.45", "$1.80 per share"],
        "ground_truth_answer": "The Board authorized a $2.0 billion share repurchase program and a quarterly dividend of $0.45 per share.",
    },
    {
        "id": "eval_20",
        "question": "What progress did TechCorp make on its ESG water and emissions goals?",
        "document": "techcorp_annual_report_2025.txt",
        "expected_page": 1,
        "reference_keywords": ["28% from our 2020 baseline", "liquid cooling", "1.2 billion gallons", "Scope 1 and Scope 2"],
        "ground_truth_answer": "TechCorp reduced Scope 1 & 2 emissions by 28% from 2020 baseline and recycled 1.2 billion gallons of water via closed-loop liquid cooling.",
    },
]


def load_benchmark_samples() -> List[EvalSample]:
    return [
        EvalSample(
            id=s["id"],
            question=s["question"],
            document=s["document"],
            expected_page=s["expected_page"],
            reference_keywords=s["reference_keywords"],
            ground_truth_answer=s["ground_truth_answer"],
        )
        for s in BENCHMARK_DATASET
    ]
