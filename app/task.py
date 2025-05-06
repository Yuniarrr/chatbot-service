from redis import Redis
from rq import Queue
from asgiref.sync import async_to_sync

from app.env import REDIS_HOST, REDIS_PORT
from app.retrieval.embed import embedding_service
from app.retrieval.vector_store import vector_store_service
from app.services.file import file_service
from app.models.files import FileStatus, FileUpdateModel

queue = Queue(connection=Redis(host=REDIS_HOST, port=REDIS_PORT))


def process_uploaded_file(
    file_id: str,
    file_name: str,
    content_type: str,
    file_path: str,
    collection_name: str,
):
    try:
        print("redis process_uploaded_file")
        loader_document = embedding_service.loader(file_name, content_type, file_path)
        splitted_document = embedding_service.split_document(loader_document)
        enriched_document = embedding_service.add_addtional_data_to_docs(
            docs=splitted_document,
            file_id=file_id,
            file_name=file_name,
        )
        vector_store_service.add_vectostore(enriched_document, collection_name)

        async_to_sync(file_service.update_file_by_id)(
            file_id, FileUpdateModel(status=FileStatus.SUCCESS)
        )
    except Exception as e:
        print(f"error in process_uploaded_file: {e}")
        async_to_sync(file_service.update_file_by_id)(
            file_id, FileUpdateModel(status=FileStatus.FAILED)
        )
        raise e
