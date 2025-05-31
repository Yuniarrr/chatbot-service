import asyncio
import sys
import time
import json
import pandas as pd
import uvicorn

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity,
)
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.messages import ToolMessage
from datasets import Dataset as HFDataset
from ragas import evaluate

from app.retrieval.vector_store import vector_store_service
from app.core.database import session_manager, pgvector_session_manager
from app.env import DATABASE_URL, PARENT_DIR
from app.retrieval.chain import chain_service


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def setup_environment():
    await session_manager.initialize()
    vector_store_service.initialize_embedding_model(
        f"{PARENT_DIR}\data\sentence-transformers\paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_store_service.initialize_pg_vector()
    chain_service.load_model_collection()
    await pgvector_session_manager.initialize()

    DB_URI = f"postgresql://{DATABASE_URL}?sslmode=disable"
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }

    pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, kwargs=connection_kwargs)
    await pool.__aenter__()  # manually enter the async context

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    chain_service.set_checkpointer(checkpointer)

    return pool  # Keep reference to close later


async def main():
    await setup_environment()

    with open(rf"{PARENT_DIR}\notebooks\question.json", "r", encoding="utf-8") as f:
        question_data = json.load(f)

    with open(
        rf"{PARENT_DIR}/notebooks/evaluate/prompting/skenario_3/skenario_3.txt",
        "r",
        encoding="utf-8",
    ) as f:
        prompt = f.read()

    print("prompt")
    print(prompt)

    question_data = question_data[:50]
    questions = [item["question"] for item in question_data]
    expected_answers = {item["question"]: item["answer"] for item in question_data}

    evaluation_data = []
    processing_times = []
    agent = chain_service.create_agent(model="openai", custom_prompt=prompt)

    count = 1

    for q in questions:
        print(f"Processing question: {count}. {q}")
        start_time = time.perf_counter()
        try:
            result = await agent.ainvoke(
                {
                    "messages": [
                        SystemMessage(
                            content="User ID atau sender pesan adalah: user-test"
                        ),
                        HumanMessage(content=str(q)),
                    ],
                }
            )

            tool_messages = [
                message
                for message in result["messages"]
                if isinstance(message, ToolMessage)
            ]

            contexts = [tool_messages[-1].content] if tool_messages else []

            ai_messages = [
                message.content
                for message in result["messages"]
                if isinstance(message, AIMessage) and message.content.strip()
            ]

            answer = (
                ai_messages[-1]
                if ai_messages
                else "Terjadi kesalahan, tidak ada respon dari AI. Tolong hubungi developer."
            )

            ground_truth = expected_answers.get(q, "No expected answer provided")

            evaluation_data.append(
                {
                    "question": q,
                    "contexts": contexts,
                    "response": answer,
                    "ground_truth": ground_truth,
                }
            )

            count = count + 1
        except Exception as e:
            print(f"Error processing question '{q}': {e}\n")

        elapsed_time = time.perf_counter() - start_time
        # processing_times.append(elapsed_time)
        processing_times.append({"question": q, "processing_time_sec": elapsed_time})
        await asyncio.sleep(10.0)

    ragas_dataset = HFDataset.from_list(evaluation_data)

    await asyncio.sleep(10.0)

    result = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            answer_correctness,
            answer_similarity,
        ],
    )

    df = result.to_pandas()
    # df["processing_time_sec"] = processing_times

    df.to_csv(
        f"{PARENT_DIR}/notebooks/evaluate/prompting/skenario_3/evaluation_result_skenario_3.csv",
        index=False,
    )
    df.to_excel(
        f"{PARENT_DIR}/notebooks/evaluate/prompting/skenario_3/evaluation_result_skenario_3.xlsx",
        index=False,
    )

    pd.DataFrame(evaluation_data).to_csv(
        f"{PARENT_DIR}/notebooks/evaluate/prompting/skenario_3/raw_evaluation_data_skenario_3.csv",
        index=False,
    )

    average_scores = df.mean(numeric_only=True)
    average_scores.to_csv(
        f"{PARENT_DIR}/notebooks/evaluate/prompting/skenario_3/average_scores_skenario_3.csv"
    )
    print("\nRata-rata setiap metrik:")
    print(average_scores)

    processing_df = pd.DataFrame(processing_times)
    df = df.merge(processing_df, left_on="user_input", right_on="question", how="left")
    df.drop(columns=["question"], inplace=True)
    df.mean(numeric_only=True)
    df.to_csv(
        f"{PARENT_DIR}/notebooks/evaluate/prompting/skenario_3/evaluation_result_skenario_3_with_time.csv",
        index=False,
    )
    average_scores = df.mean(numeric_only=True)
    average_scores.to_csv(
        f"{PARENT_DIR}/notebooks/evaluate/prompting/skenario_3/average_scores_skenario_3_with_time.csv"
    )


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
