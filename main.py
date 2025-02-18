import utils
from absl import flags
from absl import app
import clip
import numpy as np
import mmd
from datasets import load_dataset, load_from_disk

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)

def compute_cmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):
    """Calculates the CMMD distance between reference and eval image sets.

    Args:
      ref_dir: Path to the directory containing reference images.
      eval_dir: Path to the directory containing images to be evaluated.
      ref_embed_file: Path to the pre-computed embedding file for the reference images.
      batch_size: Batch size used in the CLIP embedding calculation.
      max_count: Maximum number of images to use from each directory. A
        non-positive value reads all images available except for the images
        dropped due to batching.

    Returns:
      The CMMD value between the image sets.
    """
    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = clip.ClipEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = utils.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    eval_embs = utils.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    distance = mmd.MMD(ref_embs, eval_embs, sigma=10, scale=1000)
    val = distance.compute()
    return val.numpy()

def main(argv):
    # if len(argv) != 3:
    #     raise app.UsageError("Too few/too many command-line arguments.")
    # _, dir1, dir2 = argv
    print(argv)
    
    _, dir1, dir2 = argv
    dir1 = load_from_disk("ref_images/train")
    print(
        "The CMMD value is: "
        f" {compute_cmmd(dir1, dir2, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value)}"
    )


if __name__ == "__main__":
    app.run(main)

    # ds = load_dataset("sayakpaul/coco-30-val-2014")
    # ds.save_to_disk("./ref_images")