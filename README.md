# OCR Text Extractor

A modern web application that extracts text from images using multiple AI models (OpenAI, Gemma, LLaMA, and LLaVA) and compares their performance.


## Features

- 🖼️ Upload and preview images
- 📝 Extract text using multiple AI models:
  - OpenAI GPT-4o
  - Gemma
  - LLaMA
  - LLaVA
- 🔍 Automatic comparison and ranking of results
- 💫 Real-time processing status
- 📱 Responsive design for all devices

## Tech Stack

### Frontend
- React 18
- TypeScript
- Vite
- Axios for API calls
- Modern CSS with Flexbox

### Backend
- FastAPI
- Python 3.10+
- LiteLLM for AI model integration
- python-dotenv for environment variables
- Logging for debugging

## Installation

### Prerequisites
- Node.js 16+
- Python 3.10+
- OpenAI API key
- Ollama installed locally for Gemma, LLaMA, and LLaVA models

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/ajay-4639/ajay-ocr.git
cd ajay-ocr/backend
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

5. Start the backend server:
```bash
uvicorn ocr:app --reload
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd ../frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run start-win
```

## Usage

1. Open your browser and navigate to `http://localhost:4173`
2. Click "Choose an image" or drag and drop an image
3. Click "Extract Text" to process the image
4. View the results from different AI models, ranked by accuracy

## API Endpoints

### `POST /upload-ocr`
- Accepts image file in form-data
- Returns extracted text from all models with ranking

## Environment Variables

### Backend
- `OPENAI_API_KEY`: Your OpenAI API key

## Development

### Running Tests
```bash
# Frontend
npm run test

# Backend
pytest
```

### Building for Production
```bash
# Frontend
npm run build

# Backend remains the same
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT-4V API
- Ollama for local AI model hosting
- LiteLLM for unified AI model interface
