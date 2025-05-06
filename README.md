# LangChain OpenAI Agent

## Overview
This project demonstrates the use of LangChain and OpenAI's GPT models to create an intelligent agent capable of text classification, entity extraction, and summarization. 
The agent is built using a state graph workflow to coordinate these tasks.

## Features
- **Text Classification**: Classifies text into predefined categories (News, Blog, Research, Other).
- **Entity Extraction**: Extracts named entities (Person, Organization, Location) from text.
- **Summarization**: Summarizes text into one short sentence.

## Installation
1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your OpenAI API key and model details:
        ```env
        API_HOST=github
        GITHUB_MODEL=gpt-4o-mini
        GITHUB_TOKEN=your_openai_api_key
        ```

## Usage

1. Test the agent with sample text:
    ```python
    sample_text = """
    Anthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.
    """
    state_input = {"text": sample_text}
    result = app.invoke(state_input)
    print("Classification:", result["classification"])
    print("\nEntities:", result["entities"])
    print("\nSummary:", result["summary"])
    ```

## Code Structure
- **Medium Articles Analyzer.py**: Main script to run the agent.
- **classification_node**: Function to classify text.
- **entity_extraction_node**: Function to extract entities.
- **summarize_node**: Function to summarize text.
- **StateGraph**: Workflow management for coordinating tasks.

## Contributing
Feel free to open issues or submit pull requests for improvements.


## Contact
For any inquiries, please contact paulos.marutha@gmail.com.

