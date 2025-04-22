import { ChangeEvent, FC, useState } from "react";
import axios from "axios";
import './App.css';

interface ResultDisplayProps {
  results: any;
  previews: string[];
}

const BoundingBoxOverlay: FC<{
  elements: any[];
  imageWidth: number;
  imageHeight: number;
}> = ({ elements }) => {
  const [hoveredBox, setHoveredBox] = useState<number | null>(null);

  return (
    <div className="bounding-box-overlay">
      {elements.map((el, index) => {
        const [x1, y1, x2, y2] = el.bbox;
        return (
          <div
            key={index}
            className={`bounding-box ${el.type}`}
            style={{
              position: 'absolute',
              left: `${x1 * 100}%`,
              top: `${y1 * 100}%`,
              width: `${(x2 - x1) * 100}%`,
              height: `${(y2 - y1) * 100}%`,
              opacity: hoveredBox === index ? 0.8 : 0.4
            }}
            data-type={el.type}
            title={`${el.text} (${Math.round(el.confidence * 100)}% confidence)`}
            onMouseEnter={() => setHoveredBox(index)}
            onMouseLeave={() => setHoveredBox(null)}
          />
        );
      })}
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
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content">
        <img
          src={imageUrl}
          alt={`Page ${pageNumber} of ${totalPages}`}
          className={`modal-image ${isZoomed ? 'zoomed' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            setIsZoomed(!isZoomed);
          }}
        />
        <div className="page-indicator">
          Page {pageNumber} of {totalPages}
        </div>
      </div>
    </div>
  );
};

const ResultDisplay: FC<ResultDisplayProps> = ({ results, previews }) => {
  return (
    <div className="results-content">
      <div className="results-header">
        <h3>OCR Results ({results.total_pages} {results.total_pages === 1 ? 'page' : 'pages'})</h3>
        <div className="stats">
          <span className="processing-time">{results.processing_time_seconds.toFixed(2)}s</span>
          <span className="total-cost">${results.total_cost.toFixed(6)}</span>
        </div>
      </div>

      {results.results.map((result: any, index: number) => (
        <div key={index} className="page-result">
          <div className="page-header">
            <h4>Page {result.page}</h4>
          </div>

          <div className="result-container">
            {/* Left side - Document Image */}
            <div className="image-section">
              <div className="image-wrapper">
                <img
                  src={previews[index]?.startsWith('data:') ? 
                       previews[index] : 
                       `data:image/jpeg;base64,${previews[index]}`}
                  alt={`Page ${index + 1}`}
                  className="preview-image"
                />
              </div>
            </div>

            {/* Right side - Text Results */}
            <div className="text-section">
              <div className="text-results">
                <h5>Extracted Text</h5>
                {result.elements?.map((el: any, idx: number) => (
                  <div key={idx} className="element-item">
                    <div className="text-input">
                      {el.value || el.text}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previews, setPreviews] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extractedText, setExtractedText] = useState<any | null>(null);

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
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSelectedFile(file);
      setExtractedText(null);
      setPreviews([]);
      setError(null);

      if (file.type.startsWith('image/')) {
        const previewUrl = URL.createObjectURL(file);
        setPreviews([previewUrl]);
      } else if (file.type === 'application/pdf') {
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

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await axios.post("/api/upload-ocr", formData, {
        headers: { "Content-Type": "multipart/form-data" }
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
        <p>Extract text from images using AI</p>
      </header>

      <main>
        <section className="upload-section">
          <div className="upload-box">
            <input
              type="file"
              accept="image/jpeg,image/png,image/gif,image/bmp,image/tiff,image/webp,application/pdf"
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
              <div className="file-preview">
                <p className="selected-file">
                  Selected: {selectedFile.name}
                  {selectedFile.type === 'application/pdf' && " (PDF file)"}
                </p>
                {previews.length > 0 && (
                  <div className="preview-container">
                    {previews.map((preview, index) => (
                      <div key={index} className="preview-item">
                        <img
                          src={
                            preview.startsWith('data:') || preview.startsWith('blob:')
                              ? preview
                              : `data:image/jpeg;base64,${preview}`
                          }
                          alt={`Preview page ${index + 1}`}
                          className="preview-thumbnail"
                        />
                        {previews.length > 1 && (
                          <span className="preview-page-number">Page {index + 1}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
          {error && <p className="error">{error}</p>}
          {loading && <p className="processing">Processing your file...</p>}
          <div className="preview">
            {!error && !loading && extractedText && (
              <ResultDisplay results={extractedText} previews={previews} />
            )}
          </div>
        </section>
      </main>
      <footer>
        <p>&copy; 2025 OCR Text Extractor. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;