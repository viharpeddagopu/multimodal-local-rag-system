import json

from src.embedder import LocalEmbedder
from src.vector_store import FaissStore
from src.retriever import Retriever

from src.config import FAISS_INDEX_DIR


# =========================================================
# LOAD TEST QUERIES
# =========================================================

with open(
    "evaluation/test_queries.json",
    "r",
    encoding="utf-8"
) as f:

    test_queries = json.load(f)


# =========================================================
# LOAD RETRIEVER
# =========================================================

embedder = LocalEmbedder()

vector_store = FaissStore.load(
    FAISS_INDEX_DIR
)

retriever = Retriever(
    vector_store=vector_store,
    embedder=embedder,
    use_reranker=True
)


# =========================================================
# EVALUATION
# =========================================================

results = []

correct = 0

for test in test_queries:

    query = test["query"]

    expected_document = (
        test["expected_document"]
    )

    retrieval_results = retriever.retrieve(
        query
    )

    top_chunk = retrieval_results[0]["chunk"]

    predicted_document = (
        top_chunk.document
    )

    is_correct = (
        predicted_document
        ==
        expected_document
    )

    if is_correct:

        correct += 1

    result = {

        "query": query,

        "expected_document":
            expected_document,

        "predicted_document":
            predicted_document,

        "correct":
            is_correct
    }

    results.append(result)

    print("\n===================================")

    print(f"Query: {query}")

    print(
        f"Expected: {expected_document}"
    )

    print(
        f"Predicted: {predicted_document}"
    )

    print(
        f"Correct: {is_correct}"
    )


# =========================================================
# FINAL METRICS
# =========================================================

accuracy = (
    correct
    /
    len(test_queries)
)

summary = {

    "accuracy": accuracy,

    "total_queries":
        len(test_queries),

    "correct_predictions":
        correct,

    "results": results
}


# =========================================================
# SAVE RESULTS
# =========================================================

with open(
    "evaluation/results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\n===================================")

print(
    f"Final Accuracy: "
    f"{accuracy:.2f}"
)