# Vapi Integration with Custom LLM

Welcome to the Vapi CustomLLM sample project! This guide will walk you through integrating Vapi with your custom language model.

## Getting Started

To set up this project:

1. **Create a Python Virtual Environment**
    
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    
    ```
    
2. **Set Up Environment Variables**
    - Create a `.env` file in your repository:
        
        ```bash
        cp .env.example .env
        ```
        
    - Add your `OPENAI_API_KEY` to the `.env` file.
3. **Install Dependencies**
    
    ```bash
    make install
    
    ```
    
4. **Run the FastAPI Server**
    
    ```bash
    make dev
    
    ```
    

### Exposing Localhost to the Internet

To make your local server accessible over the internet, you can use ngrok. Follow these steps:

1. **Install ngrok**
2. **Expose Localhost**
    
```bash
    ngrok http 8000   
```
    

## Conclusion

This sample project demonstrates how to integrate a custom language model with Vapi. By following these steps, developers can create a more versatile and responsive voice assistant tailored to their users' unique needs.

For more help and detailed documentation, please refer to the official [Vapi documentation](https://docs.vapi.ai/).
