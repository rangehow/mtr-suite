


import datasets
from datasketch import MinHash, MinHashLSH
import multiprocessing
from tqdm.auto import tqdm # Use auto for better notebook/script detection
import re
from functools import partial
import time # Optional: for timing

# Helper function for character n-grams (shingles)
def get_shingles(text, n=5):
    """Creates character n-grams (shingles) for a given text."""
    # Simple whitespace normalization and lowercasing
    text = re.sub(r'\s+', ' ', text).lower()
    # Generate n-grams
    return set(text[i:i+n] for i in range(len(text) - n + 1))

# Function to compute MinHash for one document (for multiprocessing)
def compute_minhash(args, n_gram_size, num_perm, column):
    """Computes MinHash for a single document item."""
    index, item = args
    try:
        text = item[column]
        # Handle empty or non-string text gracefully
        if not isinstance(text, str) or not text:
             return (index, None) # Return None MinHash for invalid text

        shingles = get_shingles(text, n=n_gram_size)
        # Handle cases where shingling results in nothing (e.g., very short text)
        if not shingles:
            return (index, None)

        m = MinHash(num_perm=num_perm)
        for shingle in shingles:
            m.update(shingle.encode('utf8'))
        return (index, m)
    except Exception as e:
        # Optional: Log the error for debugging
        # print(f"Error processing index {index}: {e}")
        return (index, None) # Indicate failure for this item


def dedup(
    dataset: datasets.Dataset,
    column: str = 'text',
    threshold: float = 0.5, # Default threshold (adjust based on needs)
    num_perm: int = 128,
    n_gram_size: int = 5   # Default n-gram size for shingles
) -> datasets.Dataset:
    """
    Deduplicates a datasets.Dataset based on text similarity using MinHashLSH.

    Args:
        dataset: The input dataset to deduplicate.
        column: The name of the text column to use for comparison.
        threshold: The Jaccard similarity threshold for documents to be considered duplicates.
                   (Higher threshold means documents must be more similar).
        num_perm: The number of permutation functions used by MinHash.
        n_gram_size: The size of character n-grams (shingles) to use.

    Returns:
        A new datasets.Dataset containing only the unique documents.
    """
    start_time = time.time()

    if column not in dataset.column_names:
         raise ValueError(f"Column '{column}' not found in dataset. Available columns: {dataset.column_names}")

    print(f"Starting deduplication process...")
    print(f"Parameters: column='{column}', threshold={threshold}, num_perm={num_perm}, n_gram_size={n_gram_size}")

    # --- 1. Compute MinHashes in Parallel ---
    print(f"Calculating MinHashes for {len(dataset)} documents...")
    minhashes = {} # Store index -> minhash mapping

    # Use functools.partial to pass fixed arguments to the worker function
    worker_func = partial(compute_minhash, n_gram_size=n_gram_size, num_perm=num_perm, column=column)

    # Compute MinHashes in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(worker_func, enumerate(dataset), chunksize=10000),
            total=len(dataset),
            desc="Computing MinHashes"
        ))

    # Populate minhashes dictionary, skipping None values (from errors or empty text)
    valid_hashes_count = 0
    skipped_count = 0
    for index, minhash_obj in results:
        if minhash_obj:
            minhashes[index] = minhash_obj
            valid_hashes_count += 1
        else:
            skipped_count +=1
            # Optional: Log skipped items
            # print(f"Skipping index {index} due to invalid text or processing error.")

    if valid_hashes_count == 0:
         print("Warning: No valid MinHashes were computed (all items might be empty or failed processing). Returning original dataset.")
         return dataset

    print(f"Computed {valid_hashes_count} valid MinHashes. Skipped {skipped_count} items.")

    # --- 2. Build LSH Index ---
    print(f"Building LSH index with threshold {threshold}...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Use a separate loop for indexing for clearer progress, though it could be combined
    for index, minhash_obj in tqdm(minhashes.items(), desc="Indexing MinHashes"):
        lsh.insert(index, minhash_obj) # Use original dataset index as the key

    # --- 3. Query LSH Index and Find Duplicates ---
    print("Querying LSH index to identify duplicate clusters...")
    seen_indices = set()
    indices_to_keep = []
    duplicate_clusters = [] # Optional: to store groups of duplicates

    # Iterate through the *original* indices for which we have hashes
    # Processing in index order helps keep the first encountered item of a cluster
    for index in tqdm(range(len(dataset)), desc="Finding duplicates"):
         # Skip if we don't have a hash (it was skipped earlier) or if already seen
         if index not in minhashes or index in seen_indices:
             continue

         # Query for similar items
         # Note: LSH query finds *candidates*. Actual similarity might be slightly different.
         # For strict deduplication based on threshold, you might re-calculate Jaccard if needed,
         # but for most cases, LSH result is sufficient.
         duplicates = lsh.query(minhashes[index])

         # The query result `duplicates` is a list of indices similar to `index` (including `index` itself)

         # Keep the current index 'i' as the representative for this cluster
         indices_to_keep.append(index)

         # Add all indices found in this cluster (including the representative) to the seen set
         seen_indices.update(duplicates)

         # Optional: Store the cluster if you want to analyze duplicates
         if len(duplicates) > 1:
             duplicate_clusters.append(list(duplicates))


    # --- 4. Create Deduplicated Dataset ---
    num_duplicates_found = len(seen_indices) - len(indices_to_keep) # Total items marked seen minus the ones we kept
    print(f"Identified {num_duplicates_found} duplicates.")
    print(f"Selecting {len(indices_to_keep)} unique documents.")

    # Select the unique documents based on the kept indices
    deduplicated_dataset = dataset.select(indices_to_keep)

    end_time = time.time()
    print(f"Deduplication finished in {end_time - start_time:.2f} seconds.")
    print(f"Original dataset size: {len(dataset)}, Deduplicated dataset size: {len(deduplicated_dataset)}")

    # Optional: Return duplicate clusters as well
    # return deduplicated_dataset, duplicate_clusters
    return deduplicated_dataset





if __name__=='__main__':

    documents = [
        'minhash is a probabilistic data structure for estimating the similarity between datasets',
        'minhash is a probability data structure for estimating the similarity between documents',
        'minhash is a probability data structure for estimating the similarity between documents',
        'this is a completely different document that should not match with others',
        'another document that might match partially with the first one'
    ]
    documents={'text':documents}
    dataset = datasets.Dataset.from_dict(documents)

    deduplicated_ds = dedup(dataset, column='text', threshold=0.7, num_perm=128, n_gram_size=5)
    print(deduplicated_ds.to_list())