# Evaluation

The evaluation pipeline measures retrieval correctness over a small set of manually curated research-paper queries.

For each query:
- the retriever searches the FAISS vector database
- the top retrieved chunk is analyzed
- the predicted source document is compared against the expected document

## Metric

Retrieval Accuracy:

```text
correct_predictions / total_queries