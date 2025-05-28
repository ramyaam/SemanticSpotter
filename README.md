# Semantic Spotter: RAG System for Insurance Domain

## 1. Background

This project demonstrates building a **Retrieval-Augmented Generation (RAG) System** in the **insurance domain** using the [LangChain](https://python.langchain.com/docs/introduction/) framework.

## 2. Problem Statement

The goal of the project is to develop a robust generative search system that can effectively answer user queries based on information retrieved from multiple policy documents. This improves accuracy and relevance of answers while maintaining transparency and trustworthiness.

## 3. Dataset

Policy documents used as source material can be found in the [Policy+Documents](./Policy+Documents) directory.

## 4. Approach Overview

This system uses LangChain, a modular and open-source framework designed for building applications with Large Language Models (LLMs). The following steps outline our solution:

### a. Document Loading

We use `PyPDFDirectoryLoader` from LangChain to read and process PDF documents in bulk from a specified folder.

### b. Chunking with RecursiveCharacterTextSplitter

Text from the PDFs is chunked into manageable sizes using LangChain’s `RecursiveCharacterTextSplitter`, preserving semantic structure for better contextual understanding.

### c. Generating Embeddings

Text embeddings are generated using `OpenAIEmbeddings` to convert textual chunks into numerical vectors for semantic search. These embeddings support similarity matching during query time.

### d. Vector Store and ChromaDB Integration

We use **ChromaDB** as our vector store to index the generated embeddings. To enhance performance and reduce re-computation, we utilize `CacheBackedEmbeddings`.

### e. Retrievers and Search

We use a `VectorStoreRetriever` to fetch top relevant document chunks based on query similarity. The retriever uses cosine similarity between the query embedding and document vectors.

### f. Re-ranking with Cross-Encoders

To improve relevance, we apply a **re-ranking layer** using `HuggingFaceCrossEncoder` with the model `BAAI/bge-reranker-base`. This re-evaluates document-query pairs using deeper semantic understanding.

### g. RAG Chain Integration

We implement a LangChain RAG pipeline that retrieves relevant documents and feeds them into an LLM prompt
