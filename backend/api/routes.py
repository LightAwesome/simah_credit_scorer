from fastapi import FastAPI

app = FastAPI()


@app.get("/score")
def read_root():
    return {"Hello": "World"}


# @app.get("/upload")
# def read_item(file: int, q: Union[str, None] = None):
#     return {"File": file, "q": q}
