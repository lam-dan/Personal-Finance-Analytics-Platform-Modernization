# Scala vs. Python Web Frameworks: A Side-by-Side Comparison

## 1) Basic Routing (Hello World)

**Scala — Akka HTTP**
```scala
val route = pathSingleSlash {
  get { complete("Hello, Akka HTTP") }
}
```

**Python — FastAPI**
```python
@app.get("/")
def hello():
    return {"message": "Hello, FastAPI"}
```

**Description:**
Defines a root route that responds to HTTP GET requests.
- In Scala (Akka HTTP), a routing DSL (`pathSingleSlash`, `get`, `complete`) declares the path and method.
- In Python (FastAPI), decorators (`@app.get`) bind a function directly to the route.
Both automatically serialize responses (Scala as plain text here, FastAPI as JSON).

---

## 2) JSON Models (Type-safe ↔ Pydantic)

**Scala — Play Framework**
```scala
case class Item(name: String, price: BigDecimal, inStock: Boolean = true)
```

**Python — FastAPI**
```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True
```

**Description:**
Models define the data schema for request/response bodies.
- Scala uses case classes combined with JSON formatters (like Play JSON, circe, or spray-json).
- FastAPI uses Pydantic models for validation, serialization, and type conversion.
Both automatically parse JSON into strongly typed objects and return validation errors if the incoming data is invalid.

---

## 3) Async I/O (Futures ↔ async/await)

**Scala — Akka HTTP**
```scala
import scala.concurrent.Future

def fetchUser(id: String): Future[String] = ??? //...
```

**Python — FastAPI**
```python
async def fetch_user(uid: str) -> str:
    # ...
    return "user_data"
```

**Description:**
Handles non-blocking operations, such as database calls or external HTTP requests.
- Scala uses `Future` (or more advanced libraries like cats-effect `IO`) for asynchronous computations.
- Python uses `async`/`await` syntax for coroutines.
Both approaches improve scalability by freeing up threads while waiting for I/O operations to complete.

---

## 4) Streaming Responses / Server-Sent Events

**Scala — Akka Streams**
```scala
import akka.stream.scaladsl.Source
import akka.http.scaladsl.model.HttpEntity

Source.tick(...).map(s => HttpEntity.ChunkStreamPart(s))
```

**Python — FastAPI**
```python
async def event_gen():
    i = 0
    while True:
        yield f"data: tick-{i}\n\n"
        i += 1
```

**Description:**
Sends data continuously over a single HTTP connection (Server-Sent Events).
- Scala uses Akka Streams to produce a potentially infinite stream of events.
- FastAPI uses a `StreamingResponse` with an async generator function.
This is useful for real-time dashboards, log streaming, or live data feeds.

---

## 5) WebSockets

**Scala — Akka HTTP**
```scala
handleWebSocketMessages(echoFlow)
```

**Python — FastAPI**
```python
from fastapi import WebSocket

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        await ws.send_text(f"Message text was: {data}")
```

**Description:**
Enables bi-directional, full-duplex communication between the server and client.
- Scala handles WebSocket messages as Akka Stream `Flow`s.
- FastAPI uses async WebSocket handlers to manage the connection lifecycle.
Suitable for chat applications, multiplayer games, or IoT telemetry.

---

## 6) Middleware / Filters

**Scala — Play Filter**
```scala
import play.api.mvc._

def apply(next: EssentialAction) = EssentialAction { rh => ... }
```

**Python — FastAPI Middleware**
```python
from starlette.middleware.base import BaseHTTPMiddleware

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(request, call_next):
        # ...
        response = await call_next(request)
        return response
```

**Description:**
Middleware (or Filters) runs code before and after each request.
- Common uses include logging, authentication checks, adding correlation IDs, and collecting metrics.
- Scala's Play Framework uses `EssentialFilter` to wrap the request handling chain.
- FastAPI uses `BaseHTTPMiddleware` to intercept requests and modify responses.

---

## 7) Validation & Error Handling

**Scala — http4s (example)**
```scala
if (in.a < 0) F.pure(Response(status = BadRequest).withEntity("must be non-negative"))
```

**Python — FastAPI**
```python
from pydantic import BaseModel, conint

class Input(BaseModel):
    a: conint(ge=0) # Must be an integer greater than or equal to 0
```

**Description:**
Validates incoming request data and returns structured error responses.
- Scala can perform validation manually or with helper libraries integrated into its JSON parsers.
- FastAPI, through Pydantic, provides powerful built-in constraint types (e.g., `conint`, `constr`, length limits).
Both frameworks allow for custom error handling to manage business logic failures gracefully.

---

## 8) Auth (JWT / OAuth2)

**Scala — Play**
```scala
req.headers.get("Authorization") match {
    case Some(authHeader) => // verify token
    case None => // handle missing auth
}
```

**Python — FastAPI**
```python
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/items/")
async def read_items(token: str = Depends(oauth2_scheme)):
    return {"token": token}
```

**Description:**
Implements authentication and authorization, commonly using JWT tokens or OAuth2.
- Scala's Play Framework typically involves reading headers and verifying tokens manually or with dedicated libraries.
- FastAPI offers dependency injection features like `OAuth2PasswordBearer` for easily extracting and validating tokens.
Both frameworks can be integrated with external authentication systems like Keycloak or Auth0.

---

## 9) Background Work

**Scala — Akka Streams**
```scala
import akka.stream.scaladsl.Source

Source.tick(...).runForeach(...) // Example of a scheduled background task
```

**Python — FastAPI**
```python
from fastapi import BackgroundTasks

def write_log(message: str):
    with open("log.txt", mode="a") as log:
        log.write(message)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, f"notification sent to {email}")
    return {"message": "Notification sent in the background"}
```

**Description:**
Runs tasks after sending the HTTP response, avoiding blocking the request/response lifecycle.
- Scala often uses scheduled Akka Streams or Akka actors for recurring or long-running jobs.
- FastAPI provides a `BackgroundTasks` utility for queuing post-response jobs like sending emails, processing data, or writing logs.

---

## 10) HTTP Client Calls

**Scala — Akka HTTP Client**
```scala
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.HttpRequest

Http().singleRequest(HttpRequest(uri = "..."))
```

**Python — httpx**
```python
import httpx

async def query_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com")
        return response.json()
```

**Description:**
Used for service-to-service communication in a microservices architecture.
- Scala uses the Akka HTTP client API for making asynchronous HTTP requests.
- Python commonly uses `httpx` for async requests or the `requests` library for sync calls.
Both support advanced features like setting headers, query parameters, authentication, and handling streaming responses.
