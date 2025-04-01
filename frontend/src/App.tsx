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

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setExtractedText(null);
      setError(null);
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
    formData.append("image", selectedFile);

    try {
      const response = await axios.post("/api/upload-ocr", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setExtractedText(response.data);
    } catch (err) {
      setError("Failed to upload image. Please try again.");
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
          <h2>Upload Image</h2>
          <div className="upload-box">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              id="file-input"
              className="file-input"
            />
            <div className="upload-controls">
              <label htmlFor="file-input" className="upload-label">
                Choose an image
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
              <p className="selected-file">Selected: {selectedFile.name}</p>
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
              <p className="processing">Processing your image...</p>
            ) : extractedText ? (
              <div className="results-content">
                <pre>{extractedText.response}</pre>
              </div>
            ) : (
              <p className="placeholder">Text will appear here after processing...</p>
            )}
          </div>
        </section>
      </main>

      <footer>
        <p>&copy; 2024 OCR Text Extractor. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default ImageUploader;
