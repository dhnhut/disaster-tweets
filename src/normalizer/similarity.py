import os
import numpy as np
import json
import re
import faiss
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from datasketch import MinHash, MinHashLSH


def filter_duplicates_with_resume(
    df,
    text_column="tweet_text",
    chunk_size=5000,
    similarity_threshold=0.75,
    checkpoint_file="checkpoint.json",
):
    print(f"Original dataset size: {len(df)}")

    # Bag-of-ngram vectorization
    print("Vectorizing text...")
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(df[text_column])

    num_rows = tfidf_matrix.shape[0]

    indices_to_drop = set()
    # For restore
    start_processing_row = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            state = json.load(f)
            start_processing_row = state.get("last_processed_row", 0)
            indices_to_drop = set(state.get("indices_to_drop", []))
        print(
            f"Resuming from row {start_processing_row}. Previously identified {len(indices_to_drop)} duplicates."
        )
    else:
        print("Starting fresh chunked cosine similarity...")

    # Apply Chunking to avoid Out-of-Memory issues with large datasets
    for start_row in range(start_processing_row, num_rows, chunk_size):
        end_row = min(start_row + chunk_size, num_rows)

        chunk_matrix = tfidf_matrix[start_row:end_row]
        target_matrix = tfidf_matrix[start_row:]

        similarities = cosine_similarity(chunk_matrix, target_matrix)
        x_indices, y_indices = np.where(similarities > similarity_threshold)

        for x, y in zip(x_indices, y_indices):
            actual_x = start_row + x
            actual_y = start_row + y

            if actual_x < actual_y and actual_x not in indices_to_drop:
                indices_to_drop.add(int(actual_y))

        # Save State to Checkpoint file
        state = {
            "last_processed_row": int(end_row),
            "indices_to_drop": list(indices_to_drop),  # JSON doesn't support sets
        }
        # to prevent corruption if interrupted exactly during the write operation
        # => Write to a temporary file first, then rename
        temp_checkpoint = f"{checkpoint_file}.tmp"
        with open(temp_checkpoint, "w") as f:
            json.dump(state, f)
        os.replace(temp_checkpoint, checkpoint_file)

        print(
            f"Processed up to row {end_row}/{num_rows}. Identified {len(indices_to_drop)} near-duplicates in total."
        )

    df_filtered = df.drop(index=list(indices_to_drop)).reset_index(drop=True)

    # Remove checkpoint file upon successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Processing complete. Checkpoint file removed.")

    print(f"Final dataset size after near-duplicate removal: {len(df_filtered)}")
    return df_filtered


def _normalize_text(text):
    text = "" if pd.isna(text) else str(text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _character_shingles(text, shingle_size=5):
    if len(text) <= shingle_size:
        return {text} if text else set()
    return {text[i : i + shingle_size] for i in range(len(text) - shingle_size + 1)}


def _make_minhash(shingles, num_perm=64):
    mh = MinHash(num_perm=num_perm)
    if not shingles:
        mh.update(b"")
        return mh
    for shingle in shingles:
        mh.update(shingle.encode("utf-8", errors="ignore"))
    return mh


def filter_duplicates_minhash_lsh(
    df, text_column="tweet_text", similarity_threshold=0.8, num_perm=64, shingle_size=5
):
    print(f"Original dataset size: {len(df)}")
    print("Running exact duplicate pre-pass...")

    normalized_texts = df[text_column].map(_normalize_text)
    seen_exact = set()
    indices_to_drop = set()
    kept_indices = []

    for idx, text in normalized_texts.items():
        if text in seen_exact:
            indices_to_drop.add(int(idx))
        else:
            seen_exact.add(text)
            kept_indices.append(int(idx))

    print(
        f"Removed {len(indices_to_drop)} exact duplicates before approximate matching."
    )
    print("Building MinHash LSH index and scanning for near-duplicates...")

    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
    signatures = {}
    next_signature_id = 0

    for idx in kept_indices:
        if idx % 1000 == 0:
            print(f"Processing index {idx} / {len(df)}", end="\r")
        text = normalized_texts.loc[idx]
        shingles = _character_shingles(text, shingle_size=shingle_size)
        signature = _make_minhash(shingles, num_perm=num_perm)

        candidate_ids = lsh.query(signature)
        is_duplicate = False

        for candidate_id in candidate_ids:
            candidate_text = normalized_texts.loc[signatures[candidate_id]["index"]]
            candidate_shingles = signatures[candidate_id]["shingles"]
            if not shingles or not candidate_shingles:
                continue
            intersection = len(shingles & candidate_shingles)
            union = len(shingles | candidate_shingles)
            exact_jaccard = intersection / union if union else 0.0
            if exact_jaccard >= similarity_threshold:
                indices_to_drop.add(int(idx))
                is_duplicate = True
                break

        if not is_duplicate:
            signature_id = f"sig_{next_signature_id}"
            next_signature_id += 1
            signatures[signature_id] = {
                "index": int(idx),
                "shingles": shingles,
                "signature": signature,
            }
            lsh.insert(signature_id, signature)

    df_filtered = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
    print(f"Final dataset size after near-duplicate removal: {len(df_filtered)}")
    return df_filtered


def filter_duplicates_faiss(
    df,
    text_column="tweet_text",
    chunk_size=50000,
    similarity_threshold=0.75,
    embedding_model="all-MiniLM-L6-v2",
    train_sample_size=100000,
    nprobe=10,
    checkpoint_file="checkpoint.json",
):
    print(f"Original dataset size: {len(df)}")
    num_rows = len(df)

    # FASS works with Dense Vectors instead of sparse TF-IDF/BoW
    print("Loading embedding model...")
    model = SentenceTransformer(embedding_model)

    # Encode in batches. model.encode is highly optimized.
    print("Vectorizing text to dense representations...")
    embeddings = model.encode(
        df[text_column].tolist(),
        batch_size=2048,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 Normalization for Cosine Similarity
    )

    # FAISS requires float32
    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]

    print("Building FAISS Index...")
    # Number of Voronoi cells (clusters).
    # A rule of thumb is 4 * sqrt(N) to 10 * sqrt(N)
    nlist = int(4 * np.sqrt(num_rows))

    # Exact inner product index for the quantizer
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train the index on a representative sample to define the clusters
    train_sample_size = min(num_rows, train_sample_size)
    print(f"Training index on {train_sample_size} samples...")
    index.train(embeddings[:train_sample_size])

    # Add all vectors to the index
    index.add(embeddings)

    # nprobe determines how many adjacent clusters to search.
    # Higher = more accurate, slower.
    index.nprobe = nprobe

    # Checkpointing setup
    indices_to_drop = set()
    start_processing_row = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            state = json.load(f)
            start_processing_row = state.get("last_processed_row", 0)
            indices_to_drop = set(state.get("indices_to_drop", []))
        print(
            f"Resuming from row {start_processing_row}. Previously identified {len(indices_to_drop)} duplicates."
        )
    else:
        print("Starting FAISS range search...")

    # range_search with Inner Product returns matches where dot_product > threshold
    for start_row in range(start_processing_row, num_rows, chunk_size):
        end_row = min(start_row + chunk_size, num_rows)
        query_vectors = embeddings[start_row:end_row]

        # lims: array of start/end indices for the results of each query
        # D: distances (similarities)
        # I: indices of the matching vectors in the database
        lims, D, I = index.range_search(query_vectors, similarity_threshold)

        # Parse the 1D arrays returned by range_search
        for i in range(len(query_vectors)):
            actual_x = start_row + i

            # Skip if we already decided to drop this item
            if actual_x in indices_to_drop:
                continue

            # Get the slice of matches for this specific query vector
            start_idx = lims[i]
            end_idx = lims[i + 1]
            matches = I[start_idx:end_idx]

            for actual_y in matches:
                # Only drop the subsequent occurrence to keep the first one
                if actual_x < actual_y:
                    indices_to_drop.add(int(actual_y))

        # Save State to Checkpoint file
        state = {
            "last_processed_row": int(end_row),
            "indices_to_drop": list(indices_to_drop),
        }
        temp_checkpoint = f"{checkpoint_file}.tmp"
        with open(temp_checkpoint, "w") as f:
            json.dump(state, f)
        os.replace(temp_checkpoint, checkpoint_file)

        print(
            f"Processed up to row {end_row}/{num_rows}. Identified {len(indices_to_drop)} near-duplicates."
        )

    df_filtered = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
    return df_filtered
