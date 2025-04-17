import type { ChangeEvent, FC } from "react";
import { useState } from "react";
import axios from "axios";
import './App.css';

const ImageUploader: FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [extractedText, setExtractedText] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previews, setPreviews] = useState<string[]>([]);

  const generatePdfPreview = async (file: File) => {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post("/api/convert-preview", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setPreviews(response.data.pages);
    } catch (err: any) {
      console.error("Preview generation failed:", err);
      setError(err.response?.data?.detail || "Failed to generate preview");
      setPreviews([]);
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      
      const fileType = file.type;
      if (!fileType.startsWith('image/') && fileType !== 'application/pdf') {
        setError("Please upload an image or PDF file");
        return;
      }

      if (file.size > 20 * 1024 * 1024) {
        setError("File size too large. Please upload a file smaller than 20MB");
        return;
      }

      setSelectedFile(file);
      setError(null);
      setExtractedText(null);
      setPreviews([]); // Clear existing previews
      
      setLoading(true);
      
      try {
        if (fileType.startsWith('image/')) {
          const previewUrl = URL.createObjectURL(file);
          setPreviews([previewUrl]);
        } else if (fileType === 'application/pdf') {
          await generatePdfPreview(file);
        }
      } catch (err) {
        console.error("File handling error:", err);
        setError("Failed to process file");
      } finally {
        setLoading(false);
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
    formData.append("file", selectedFile);

    try {
      console.log("Sending request to server...");
      const response = await axios.post("/api/upload-ocr", formData, {
        headers: { 
          "Content-Type": "multipart/form-data"
        },
      });
      console.log("Server response:", response.data);
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
              accept="image/jpeg,image/png,image/gif,image/bmp,image.tiff,image.webp,application/pdf,image.heic,image.heif"
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
          {previews.length > 0 && (
            <div className="preview">
              <h3>Preview</h3>
              {previews.map((preview, index) => (
                <img key={index} src={preview} alt={`Preview ${index + 1}`} />
              ))}
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
              <ResultDisplay results={extractedText} previews={previews} />
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

const ImageModal: FC<{
  imageUrl: string;
  pageNumber: number;
  totalPages: number;
  onClose: () => void;
}> = ({ imageUrl, pageNumber, totalPages, onClose }) => {
  const [isZoomed, setIsZoomed] = useState(false);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsZoomed(!isZoomed);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content">
        <img
          src={imageUrl}
          alt={`Page ${pageNumber} of ${totalPages}`}
          className={`modal-image ${isZoomed ? 'zoomed' : ''}`}
          onClick={handleClick}
        />
        {totalPages > 1 && (
          <div className="page-indicator">
            Page {pageNumber} of {totalPages}
          </div>
        )}
      </div>
    </div>
  );
};

const ResultDisplay: FC<{ 
  results: any; 
  previews: string[] 
}> = ({ 
  results, 
  previews 
}) => {
  const [showModal, setShowModal] = useState(false);
  const [selectedPage, setSelectedPage] = useState(0);

  if (!results || !results.results || !Array.isArray(results.results)) {
    return <div>No results available</div>;
  }

  return (
    <div className="results-content">
      <h3>
        Extracted Text ({results.total_pages} {results.total_pages === 1 ? 'page' : 'pages'})
        <span className="processing-time">
          {' '}• Processed in {results.processing_time_seconds.toFixed(2)}s
        </span>
        <span className="total-cost"> 
          {' '}• Total Cost: ${results.total_cost.toFixed(6)}
        </span>
      </h3>
      
      {results.results.map((result: any, index: number) => (
        <div key={index} className="page-result">
          <h4>
            Page {result.page}
            <span className="page-cost">Cost: ${result.cost.toFixed(6)}</span>
          </h4>
          <div className="result-container">
            <div className="image-container">
              {previews[index] && (
                <img
                  src={previews[index].startsWith('data:') ? previews[index] : `data:image/jpeg;base64,${previews[index]}`}
                  alt={`Page ${result.page}`}
                  className="original-image"
                  onClick={() => {
                    setSelectedPage(index);
                    setShowModal(true);
                  }}
                />
              )}
            </div>
            <div className="text-output">
              <pre>{result.text}</pre>
            </div>
          </div>
        </div>
      ))}
      
      {showModal && previews[selectedPage] && (
        <ImageModal
          imageUrl={previews[selectedPage].startsWith('data:') ? previews[selectedPage] : `data:image/jpeg;base64,${previews[selectedPage]}`}
          pageNumber={selectedPage + 1}
          totalPages={results.total_pages}
          onClose={() => setShowModal(false)}
        />
      )}
    </div>
  );
};

export default ImageUploader;
