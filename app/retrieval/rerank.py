import asyncio
import operator

from typing import List
from typing import Any
from langchain.schema import BaseRetriever, Document
from langchain_openai import ChatOpenAI
from langchain_core.documents import BaseDocumentCompressor, Document
from typing import Optional, Sequence
from langchain_core.callbacks import Callbacks
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from infinity_emb import AsyncEmbeddingEngine, EngineArgs


class ManualRerankRetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, llm: ChatOpenAI, top_k: int = 5):
        self.base_retriever = base_retriever
        self.llm = llm
        self.top_k = top_k

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # Ambil kandidat dokumen dari retriever hybrid
        candidate_docs = await self.base_retriever.aget_relevant_documents(query)
        # Rerank dengan LLM
        reranked_docs = await self.rerank_documents(query, candidate_docs)
        return reranked_docs[: self.top_k]

    async def rerank_documents(
        self, query: str, docs: List[Document]
    ) -> List[Document]:
        # Buat prompt list
        prompts = [
            f"""Skor relevansi dokumen ini terhadap pertanyaan berikut dalam skala 1 (tidak relevan) sampai 10 (sangat relevan):
            Pertanyaan: {query}
            Dokumen: {doc.page_content}
            Jawab hanya dengan angka saja.""".strip()
            for doc in docs
        ]

        # Kirim ke LLM secara batch (lebih efisien)
        responses = await self.llm.abatch(prompts)

        reranked = []
        for resp, doc in zip(responses, docs):
            try:
                score = int(resp.content.strip())
            except Exception:
                score = 0
            reranked.append((score, doc))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in reranked]


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            scores = self.reranking_function.predict(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            from sentence_transformers import util

            query_embedding = self.embedding_function(query)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents]
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        docs_with_scores = list(zip(documents, scores))
        if self.r_score:
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results


# class Reranker:
#     def __init__(self, model_name="BAAI/bge-reranker-base"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

#     def predict(self, query_doc_pairs):
#         inputs = self.tokenizer(
#             [q for q, d in query_doc_pairs],
#             [d for q, d in query_doc_pairs],
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#         )
#         with torch.no_grad():
#             scores = self.model(**inputs).logits.squeeze(-1)
#             probs = F.softmax(scores, dim=0)
#         return probs.tolist()


# class Reranker:
#     def __init__(self, model_name="BAAI/bge-reranker-base", device="cpu"):
#         self.engine = AsyncEmbeddingEngine.from_args(
#             EngineArgs(
#                 model_name_or_path=model_name,
#                 device=device,
#                 engine="torch",
#                 bettertransformer=False,
#             )
#         )

#     async def rerank(self, query: str, docs: List[str]):
#         async with self.engine:
#             scores, _ = await self.engine.rerank(query=query, docs=docs)
#         return scores

#     async def predict(self, query: str, documents: List[str]):
#         return await self.rerank(query, documents)


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", device="cpu"):
        self.engine = AsyncEmbeddingEngine.from_args(
            EngineArgs(
                model_name_or_path=model_name,
                device=device,
                engine="torch",
                bettertransformer=False,
            )
        )

    async def rerank(self, query: str, docs: List[str]):
        async with self.engine:
            scores, _ = await self.engine.rerank(query=query, docs=docs)
        return scores

    def predict(self, query: str, documents: List[str]):
        return asyncio.run(self.rerank(query, documents))


class AsyncReranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    ):
        self.engine_args = EngineArgs(
            model_name_or_path=model_name,
            device=device,
            engine="torch",  # or "onnx" if compatible
            bettertransformer=False,
        )
        self.engine = AsyncEmbeddingEngine.from_args(self.engine_args)

    async def rerank(
        self, query: str, documents: list[Document], top_n: int = 5
    ) -> list[Document]:
        texts = [doc.page_content for doc in documents]
        async with self.engine:
            scores, _ = await self.engine.rerank(query=query, docs=texts)
            print(scores[0])
            print(dir(scores[0]))

        sorted_docs = sorted(
            zip(documents, scores),
            key=lambda x: getattr(x[1], "score", getattr(x[1], "value", float("-inf"))),
            reverse=True,
        )

        return [doc for doc, _ in sorted_docs[:top_n]]


class AsyncCrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )

    async def rerank(self, query: str, documents: list[str], top_n: int):
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]

        inputs = self.tokenizer(
            [q for q, d in pairs],
            [d for q, d in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(-1)

        scores = logits.cpu().tolist()

        # Urutkan dokumen berdasarkan skor (desc)
        sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in sorted_docs[:top_n]]

        return top_docs
