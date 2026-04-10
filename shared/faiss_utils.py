"""
Shared FAISS utilities for GPU index management.
Used by embedding/ and eval/ modules.
"""
import faiss
from loguru import logger


def make_vres_vdev(ngpus, gpu_resources, i0=0, i1=-1):
    """Return vectors of device ids and resources useful for gpu_multiple."""
    vres = faiss.GpuResourcesVector()
    vdev = faiss.Int32Vector()
    if i1 == -1:
        i1 = ngpus
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev


def move_index_to_gpu(index, ngpus, gpu_resources):
    """Move a FAISS index to GPU(s) if available.
    
    Args:
        index: FAISS index to move.
        ngpus: Number of available GPUs.
        gpu_resources: List of GPU resources (from faiss.StandardGpuResources).
        
    Returns:
        GPU-backed FAISS index, or original CPU index if no GPUs.
    """
    if ngpus > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        vres, vdev = make_vres_vdev(ngpus, gpu_resources)
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        logger.info("FAISS index successfully moved to GPU.")
        return gpu_index
    else:
        logger.info("No GPU available, using CPU index.")
        return index
