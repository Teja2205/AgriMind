# AgriMind 🌱

AI-powered crop disease diagnosis API. Send a photo of your crop and get instant disease analysis, pesticide recommendations, and dosage guidance — powered by GPT-4o Vision.

## What it does

AgriMind analyses crop images and returns:
- Disease identification
- Recommended pesticides
- Minimum dosage for maximum yield
- Application best practices

## Tech Stack

- Python 3.13
- FastAPI
- OpenAI GPT-4o Vision
- Pydantic
- UV (package manager)

## Setup

1. Clone the repo
```bash
   git clone https://github.com/Teja2205/AgriMind.git
   cd AgriMind
```

2. Create and activate virtual environment
```bash
   uv venv
   source .venv/bin/activate
```

3. Install dependencies
```bash
   uv pip install -r requirements.txt
```

4. Create a `.env` file in the root folder
add open ai api key


5. Run the server
```bash
   uvicorn main:app --reload
```

6. Open the API docs
http://localhost:8000/docs


## Example Request

```json
{
  "image_url": "https://your-crop-image.jpg",
  "crop": "tomato"
}
```

## Example Response

```json
{
  "diagnosis": "The tomato leaf shows symptoms consistent with powdery mildew. Recommended: Sulfur-based fungicides (3-10 lbs/acre), Potassium bicarbonate (2.5-5 tbsp/gallon), Neem oil (2 tbsp/gallon). Apply in early morning or late evening. Rotate fungicides to prevent resistance."
}
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/diagnose` | Diagnose crop disease from image |

## Author

Built by Teja as part of the AI Engineer learning journey 2026.