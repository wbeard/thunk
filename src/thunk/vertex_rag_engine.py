import os
import logging
import tempfile
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai.generative_models


logger = logging.getLogger(__name__)

logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

@dataclass
class Document:
    """Document class compatible with the original interface"""

    id: str
    title: str
    url: str
    content: str
    summary: str
    source_type: str
    date_collected: str
    relevance_score: float
    metadata: Dict[str, Any]


class VertexRagEngineAPI:
    """Interface for Vertex AI RAG Engine API to manage document corpus"""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        corpus_display_name: str = "default_corpus",
    ):
        """
        Initialize Vertex AI RAG Engine API

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region (default: us-central1)
            corpus_display_name: Name for the RAG corpus
        """
        self.project_id = project_id
        self.location = location
        self.corpus_display_name = corpus_display_name
        self.corpus = None

        vertexai.init(project=project_id, location=location)

        # Base URL for REST API calls
        self.base_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}"

        # Initialize or get existing corpus
        self._initialize_corpus()

    def _initialize_corpus(self):
        """Initialize or retrieve existing RAG corpus"""
        try:
            # Try to find existing corpus by display name
            corpora = rag.list_corpora()
            for corpus in corpora:
                if corpus.display_name == self.corpus_display_name:
                    self.corpus = corpus
                    logger.debug(f"Found existing corpus: {corpus.name}")
                    return

            # Create new corpus if not found
            embedding_model_config = rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model="publishers/google/models/text-embedding-005"
                )
            )

            print("✍️  Initializing corpus")

            self.corpus = rag.create_corpus(
                display_name=self.corpus_display_name,
                backend_config=rag.RagVectorDbConfig(
                    rag_embedding_model_config=embedding_model_config
                ),
            )

            print(f"Created new corpus: {self.corpus.name}")

        except Exception as e:
            logger.error(f"Failed to initialize corpus: {e}")
            raise

    def store_document(self, document: Document) -> str:
        """
        Store a document in the RAG corpus by creating a temporary file and importing it

        Args:
            document: Document object to store

        Returns:
            str: Document ID (file ID from Vertex AI)
        """
        if not self.corpus:
            raise ValueError("Corpus not initialized")

        temp_file_path = None
        try:
            # Create a temporary file with the document content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as temp_file:
                # Write document content with metadata as header
                content = f"""Title: {document.title}
                          URL: {document.url}
                          Summary: {document.summary}
                          Source Type: {document.source_type}
                          Date Collected: {document.date_collected}
                          Metadata: {document.metadata}

                          Content:
                          {document.content}
                          """
                temp_file.write(content)
                temp_file_path = temp_file.name

            # Upload the file to the corpus
            operation = rag.upload_file(
                corpus_name=self.corpus.name,
                path=temp_file_path,
                display_name=document.title,
                description=document.summary,
            )

            # The operation contains the file reference
            return operation.name

        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return document.id
        finally:
            os.unlink(temp_file_path) if temp_file_path else None

    def search_corpus(
        self,
        query: str,
        limit: int = 5,
        vector_distance_threshold: Optional[float] = None,
        rag_file_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Search the corpus for relevant documents using Vertex AI RAG retrieval query

        Args:
            query: Search query string
            limit: Maximum number of results to return (top_k)
            vector_distance_threshold: Optional distance threshold for filtering results
            rag_file_ids: Optional list of specific file IDs to search within

        Returns:
            List[Document]: List of relevant documents
        """
        if not self.corpus:
            logger.error("Corpus not initialized")
            return []

        try:
            # Configure retrieval settings
            rag_retrieval_config = rag.RagRetrievalConfig(top_k=limit)

            # Add distance filter if specified
            if vector_distance_threshold is not None:
                rag_retrieval_config.filter = rag.Filter(
                    vector_distance_threshold=vector_distance_threshold
                )

            # Configure RAG resource
            rag_resource = rag.RagResource(rag_corpus=self.corpus.name)

            # Add specific file IDs if provided
            if rag_file_ids:
                rag_resource.rag_file_ids = rag_file_ids

            # Perform retrieval query
            response = rag.retrieval_query(
                rag_resources=[rag_resource],
                text=query,
                rag_retrieval_config=rag_retrieval_config,
            )

            # Parse response and convert to Document objects
            documents = []

            # Access contexts from the response
            contexts = response.contexts.contexts

            for context in contexts:
                # Extract information from context
                source_uri = getattr(context, "source_uri", "")
                text = getattr(context, "text", "")
                distance = getattr(context, "distance", 0.0)

                # Parse the document content to extract metadata
                lines = text.split("\n")
                title = ""
                url = ""
                summary = ""
                source_type = ""
                date_collected = ""
                metadata = {}
                content_start_idx = 0

                # Extract metadata from header
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith("Title: "):
                        title = line[7:].strip()
                    elif line.startswith("URL: "):
                        url = line[5:].strip()
                    elif line.startswith("Summary: "):
                        summary = line[9:].strip()
                    elif line.startswith("Source Type: "):
                        source_type = line[13:].strip()
                    elif line.startswith("Date Collected: "):
                        date_collected = line[16:].strip()
                    elif line.startswith("Content:"):
                        content_start_idx = i + 1
                        break

                # Extract actual content
                content = "\n".join(lines[content_start_idx:]).strip()

                # Create Document object
                doc = Document(
                    id=source_uri,
                    title=title or "Unknown",
                    url=url,
                    content=content,
                    summary=summary,
                    source_type=source_type,
                    date_collected=date_collected,
                    relevance_score=1.0
                    - distance,  # Convert distance to relevance score
                    metadata=metadata,
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to search corpus: {e}")
            return []

    def get_corpus_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the corpus

        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not self.corpus:
            logger.error("Corpus not initialized")
            return {}

        try:
            # Get corpus details
            corpus_info = rag.get_corpus(name=self.corpus.name)

            # Get list of files in the corpus
            files = rag.list_files(corpus_name=self.corpus.name)

            file_objects = []

            for file in files:
                file_objects.append(file)

            summary = {
                "corpus_name": corpus_info.name,
                "display_name": corpus_info.display_name,
                "description": getattr(corpus_info, "description", ""),
                "create_time": getattr(corpus_info, "create_time", ""),
                "update_time": getattr(corpus_info, "update_time", ""),
                "file_count": len(file_objects),
                "files": file_objects,
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to get corpus summary: {e}")
            return {}

    def generate_with_rag(
        self, prompt: str, model_name: str = "gemini-2.5-flash-preview-05-20"
    ) -> str:
        """
        Generate content using RAG with the specified model

        Args:
            prompt: Input prompt for generation
            model_name: Name of the generative model to use

        Returns:
            str: Generated response
        """
        if not self.corpus:
            logger.error("Corpus not initialized")
            return ""

        try:
            # Create the generative model
            model = GenerativeModel(
                model_name,
                generation_config=vertexai.generative_models.GenerationConfig(
                    max_output_tokens=65536,
                    temperature=0.2,
                ),
            )

            # Create RAG retrieval tool
            rag_tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[rag.RagResource(rag_corpus=self.corpus.name)],
                        rag_retrieval_config=rag.RagRetrievalConfig(top_k=20),
                    ),
                )
            )

            # Generate content with RAG
            response = model.generate_content(prompt, tools=[rag_tool])

            return response.text

        except Exception as e:
            logger.error(f"Failed to generate with RAG: {e}")
            return ""

    def delete_corpus(self):
        """Delete the RAG corpus"""
        if self.corpus:
            try:
                rag.delete_corpus(name=self.corpus.name)
                logger.info(f"Deleted corpus: {self.corpus.name}")
                self.corpus = None
            except Exception as e:
                logger.error(f"Failed to delete corpus: {e}")

    def import_files_from_paths(
        self, file_paths: List[str], chunk_size: int = 1024, chunk_overlap: int = 200
    ):
        """
        Import files from Cloud Storage or Google Drive paths

        Args:
            file_paths: List of paths (gs:// URLs or Google Drive links)
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
        """
        if not self.corpus:
            raise ValueError("Corpus not initialized")

        try:
            # Import files to the corpus
            operation = rag.import_files(
                corpus_name=self.corpus.name,
                paths=file_paths,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            logger.info(f"Import operation started: {operation.name}")
            return operation

        except Exception as e:
            logger.error(f"Failed to import files: {e}")
            raise
