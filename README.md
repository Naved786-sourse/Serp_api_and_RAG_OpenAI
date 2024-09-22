# Serp_api_and_RAG_OpenAI

# Streamlit LangChain Query System

This repository contains a Streamlit application that handles user queries by either retrieving real-time data (via SerpAPI) or querying information from a PDF document using LangChain and FAISS.

## Features

- **Real-Time Queries**: Get real-time information like current prices or news using SerpAPI.
- **PDF Document Queries**: Retrieve relevant information from a PDF document stored in a FAISS vector database.
- **Streamlit UI**: Simple web interface to interact with the query system.

## Installation and Setup

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- OpenAI API Key and SerpAPI Key (stored in a `.env` file)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
