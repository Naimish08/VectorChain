import uvicorn

if __name__ == "__main__":
    # This command starts the Uvicorn server, pointing it to the 'app' instance
    # inside the 'app.main' module. The reload=True flag is great for development.
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )