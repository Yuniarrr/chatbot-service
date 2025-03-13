import logging
import requests
import os

from typing import Optional, Union
from huggingface_hub import snapshot_download

from app.core.logger import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    embedding_batch_size,
):
    if embedding_engine == "":
        return lambda query: embedding_function.encode(query).tolist()
    elif embedding_engine in ["ollama"]:
        func = lambda query: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            url=url,
            key=key,
        )

        def generate_multiple(query, func):
            if isinstance(query, list):
                embeddings = []
                for i in range(0, len(query), embedding_batch_size):
                    embeddings.extend(func(query[i : i + embedding_batch_size]))
                return embeddings
            else:
                return func(query)

        return lambda query: generate_multiple(query, func)


def generate_ollama_batch_embeddings(
    model: str, texts: list[str], url: str, key: str = ""
) -> Optional[list[list[float]]]:
    try:
        r = requests.post(
            f"{url}/api/generate",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={"input": texts, "model": model},
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        # print(e)
        log.error("Error generate_ollama_batch_embeddings: %s", e)
        return None


def generate_embeddings(engine: str, model: str, text: Union[str, list[str]], **kwargs):
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")

    if isinstance(text, list):
        embeddings = generate_ollama_batch_embeddings(
            **{"model": model, "texts": text, "url": url, "key": key}
        )
    else:
        embeddings = generate_ollama_batch_embeddings(
            **{"model": model, "texts": [text], "url": url, "key": key}
        )
    return embeddings[0] if isinstance(text, str) else embeddings


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model
    local_files_only = True

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model
