if __name__ == '__main__':
    import uvicorn
    uvicorn.run("my_app.endpoints_api:app", host="0.0.0.0", port=7000, reload=True)

