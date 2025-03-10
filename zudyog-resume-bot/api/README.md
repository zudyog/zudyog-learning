### To run the application:

## Install dependencies
```bash
  pip install -r requirements.txt
```
## Run application
```bash
    uvicorn src.main:app --reload --port 8101 
```

1. Build the container:
```bash
    docker-compose build
```
2. tart the services:
```bash
    docker-compose up
```
3. To stop the services:
```bash
    docker-compose down
```

