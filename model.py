import os

from huggingface_hub import snapshot_download, login
from dotenv import load_dotenv
from app.env import SENTENCE_TRANSFORMERS_HOME, RAG_EMBEDDING_MODEL

load_dotenv()
login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

print(f"============ {RAG_EMBEDDING_MODEL} ============")
snapshot_download(repo_id=RAG_EMBEDDING_MODEL, local_dir=SENTENCE_TRANSFORMERS_HOME)

# snapshot_download(
#     repo_id="indobenchmark/indobert-base-p1",
#     local_dir="data/indobenchmark/indobert-base-p1",
# )
