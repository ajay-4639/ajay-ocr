import type { ChangeEvent, FC } from "react";
import { useState } from "react";
import axios from "axios";
import './App.css';

const ImageUploader: FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [extractedText, setExtractedText] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const generatePdfPreview = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post("/api/convert-preview", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        responseType: 'blob'
      });

      const previewUrl = URL.createObjectURL(response.data);
      setPreviewUrl(previewUrl);
    } catch (err) {
      console.error("Failed to generate PDF preview:", err);
      setError("Failed to generate PDF preview");
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      
      // Check file type
      const fileType = file.type;
      if (!fileType.startsWith('image/') && fileType !== 'application/pdf') {
        setError("Please upload an image or PDF file");
        return;
      }

      setSelectedFile(file);
      setError(null);
      setExtractedText(null);
      
      // Handle preview generation
      if (fileType.startsWith('image/')) {
        setPreviewUrl(URL.createObjectURL(file));
      } else if (fileType === 'application/pdf') {
        setPreviewUrl(null); // Clear existing preview
        await generatePdfPreview(file);
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    // Change 'image' to 'file' to match backend expectation
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("/api/upload-ocr", formData, {
        headers: { 
          "Content-Type": "multipart/form-data"
        },
      });
      setExtractedText(response.data);
    } catch (err: any) {
      console.error("Upload error:", err);
      setError(err.response?.data?.detail || "Failed to upload file. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>OCR Text Extractor</h1>
        <p>Extract text from images using multiple AI models</p>
      </header>

      <main>
        <section className="upload-section">
          <h2>Upload Image or PDF</h2>
          <div className="upload-box">
            <input
              type="file"
              accept="image/jpeg,image/png,image/gif,image/bmp,image/tiff,image/webp,application/pdf,image.heic,image.heif"
              onChange={handleFileChange}
              id="file-input"
              className="file-input"
            />
            <div className="upload-controls">
              <label htmlFor="file-input" className="upload-label">
                Choose a file
              </label>
              <button
                onClick={handleUpload}
                disabled={loading || !selectedFile}
                className="upload-button"
              >
                {loading ? "Processing..." : "Extract Text"}
              </button>
            </div>
            {selectedFile && (
              <p className="selected-file">
                Selected: {selectedFile.name}
                {selectedFile.type === 'application/pdf' && " (PDF file)"}
              </p>
            )}
          </div>
          {previewUrl && (
            <div className="preview">
              <h3>Preview</h3>
              <img src={previewUrl} alt="Preview" />
            </div>
          )}
        </section>

        <section className="result-section">
          <h2>Extracted Text Results</h2>
          <div className="result-box">
            {error ? (
              <p className="error">{error}</p>
            ) : loading ? (
              <p className="processing">Processing your file...</p>
            ) : extractedText ? (
              <ResultDisplay results={extractedText} previewUrl={previewUrl} />
            ) : (
              <p className="placeholder">Text will appear here after processing...</p>
            )}
          </div>
        </section>
      </main>

      <footer>
        <p>&copy; 2025 OCR Text Extractor. All rights reserved.</p>
      </footer>
    </div>
  );
};

const ResultDisplay: FC<{ results: any, previewUrl: string | null }> = ({ results, previewUrl }) => {
  return (
    <div className="results-content">
      <h3>
        Extracted Text ({results.total_pages} {results.total_pages === 1 ? 'page' : 'pages'})
        <span className="processing-time">
          {' '}• Processed in {results.processing_time_seconds.toFixed(2)}s
        </span>
      </h3>
      
      {results.results.map((result: any, index: number) => (
        <div key={index} className="page-result">
          <h4>Page {result.page}</h4>
          <div className="result-container">
            <div className="image-container">
              {previewUrl && (
                <img src={previewUrl} alt="Original" className="original-image" />
              )}
            </div>
            <div className="model-outputs">
              <div className="model-output">
                <h5>
                  <span className="model-icon">🤖</span> 
                  OpenAI Analysis
                </h5>
                <pre>{formatText(result.openai_output)}</pre>
              </div>
              <div className="model-output">
                <h5>
                  <span className="model-icon">📝</span> 
                  Gemini Analysis
                </h5>
                <pre>{formatText(result.gemini_output)}</pre>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

// Helper function to format text
const formatText = (text: string): string => {
  // Remove extra whitespace and normalize line breaks
  return text
    .trim()
    .replace(/\n{3,}/g, '\n\n') // Replace multiple line breaks with double line break
    .replace(/\s{2,}/g, ' '); // Replace multiple spaces with single space
};

export default ImageUploader;
