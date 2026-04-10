"""
Shared embedding utilities using infinity_emb.
Used by embedding/, eval/, and eval/rewrite/ modules.
"""
import asyncio
import numpy as np
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine


def split_sentences(sentences, engine_count):
    """Split sentences into `engine_count` approximately equal chunks."""
    avg_len = len(sentences) // engine_count
    chunks = [sentences[i * avg_len: (i + 1) * avg_len] for i in range(engine_count)]
    # If there are leftovers, add them to the last chunk
    if len(sentences) % engine_count != 0:
        chunks[-1].extend(sentences[engine_count * avg_len:])
    return chunks


async def embed_text(engine: AsyncEmbeddingEngine, sentences_chunk: list):
    """Embed a chunk of sentences using an async embedding engine."""
    async with engine:
        embeddings, _ = await engine.embed(sentences=sentences_chunk)
    return embeddings


async def run_parallel_embeddings(array, sentences, engine_count):
    """Run embedding computation in parallel across multiple engines."""
    sentence_chunks = split_sentences(sentences, engine_count)

    tasks = []
    for i in range(engine_count):
        tasks.append(embed_text(array[i], sentence_chunks[i]))

    results = await asyncio.gather(*tasks)
    combined_embeddings = np.concatenate([result for result in results], axis=0)
    return combined_embeddings
