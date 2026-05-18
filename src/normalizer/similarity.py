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

def vectorize_faiss(
    df, text_column="tweet_text", model_name="all-MiniLM-L6-v2", batch_size=2048
):
    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print("Vectorizing text to dense representations...")
    embeddings = model.encode(
        df[text_column].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 Normalization for Cosine Similarity
    )
    return np.array(embeddings, dtype=np.float32)

def train_faiss_index(
    embeddings,
    nlist=None,
    train_sample_size=100000,
    nprobe=10,
    use_gpu=True,
):
    print(f"Building FAISS Index on CPU...")

    num_rows = embeddings.shape[0]
    dimension = embeddings.shape[1]
    nlist = int(4 * np.sqrt(num_rows)) if nlist is None else nlist

    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    gpu_index = None
    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        print("Transferring index to GPU for training...")
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        train_index = gpu_index
    else:
        train_index = index
    train_sample_size = min(num_rows, train_sample_size)
    print(f"Training index on {train_sample_size} samples...")
    train_index.train(embeddings[:train_sample_size])
    train_index.add(embeddings)
    train_index.nprobe = nprobe
    return train_index, gpu_index


def faiss_range_search(
    embeddings,
    train_index,
    gpu_index=None,
    chunk_size=50000,
    similarity_threshold=0.75,
    checkpoint_file="checkpoint.json",
):
    num_rows = len(embeddings)

    # range_search only runs on CPU
    if gpu_index is not None:
        print("Converting index back to CPU for range search...")
        index = faiss.index_gpu_to_cpu(gpu_index)
    else:
        index = train_index

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
        print("Starting new FAISS radius search...")

    for start_row in range(start_processing_row, num_rows, chunk_size):
        end_row = min(start_row + chunk_size, num_rows)
        query_vectors = embeddings[start_row:end_row]

        # range_search with Inner Product returns matches where dot_product > threshold
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

    print("Remove checkpoint file upon successful completion.")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return indices_to_drop


def faiss_neighbors_search(
    embeddings,
    train_index,
    gpu_index=None,
    chunk_size=50000,
    top_k=20,
    similarity_threshold=0.75,
    checkpoint_file="checkpoint.json",
):
    num_rows = len(embeddings)

    # search can use GPU when available
    index = gpu_index if gpu_index is not None else train_index

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
        print("Starting new FAISS knearest search...")

    for start_row in range(start_processing_row, num_rows, chunk_size):
        end_row = min(start_row + chunk_size, num_rows)
        query_vectors = embeddings[start_row:end_row]

        # search returns top_k neighbors per query, then we filter by threshold
        distances, matches = index.search(query_vectors, top_k)

        for i in range(len(query_vectors)):
            actual_x = start_row + i

            if actual_x in indices_to_drop:
                continue

            for actual_y, score in zip(matches[i], distances[i]):
                # -1 can appear when fewer than top_k neighbors are found
                if actual_y < 0 or score < similarity_threshold:
                    continue
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

    print("Remove checkpoint file upon successful completion.")
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return indices_to_drop

def filter_duplicates_faiss(
    df,
    text_column="tweet_text",
    nlist=None,
    chunk_size=50000,
    embedding_model="all-MiniLM-L6-v2",
    train_sample_size=100000,
    nprobe=10,
    search_type="radius",  # "radius" or "knearest"
    top_k=20,
    similarity_threshold=0.75,
    checkpoint_file="checkpoint.json",
):
    print(f"Original dataset size: {len(df)}")

    embeddings = vectorize_faiss(
        df, text_column=text_column, model_name=embedding_model
    )
    train_index, gpu_index = train_faiss_index(
        embeddings,
        nlist=nlist,
        train_sample_size=train_sample_size,
        nprobe=nprobe,
    )
    if search_type == "radius":
        indices_to_drop = faiss_range_search(
            embeddings,
            train_index,
            gpu_index=gpu_index,
            chunk_size=chunk_size,
            similarity_threshold=similarity_threshold,
            checkpoint_file=checkpoint_file,
        )
    elif search_type == "knearest":
        indices_to_drop = faiss_neighbors_search(
            embeddings,
            train_index,
            gpu_index=gpu_index,
            chunk_size=chunk_size,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            checkpoint_file=checkpoint_file,
        )
    else:
        raise ValueError("search_type must be 'radius' or 'knearest'")

    df_filtered = df.drop(index=list(indices_to_drop)).reset_index(drop=True)
    return df_filtered
