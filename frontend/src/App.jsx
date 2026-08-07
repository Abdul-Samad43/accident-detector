import { useState } from "react"

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  
  function handleFileChange(e) {
    const selected = e.target.files[0]
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
    setResult(null)
  }

  
  async function handleDetect() {
    if (!file) return
    setLoading(true)

    const formData = new FormData()
    formData.append("file", file)

    const response = await fetch("http://127.0.0.1:8000/detect", {
      method: "POST",
      body: formData,
    })

    const data = await response.json()
    setResult(data)
    setLoading(false)
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0f0f0f",
      color: "white",
      fontFamily: "Arial",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      padding: "40px"
    }}>

      {/* Header */}
      <h1 style={{ fontSize: "2.5rem", marginBottom: "5px" }}>
        🚨 Accident Detector
      </h1>
      <p style={{ color: "#888", marginBottom: "40px" }}>
        AI powered — YOLO Object Detection
      </p>

      
      <label style={{
        border: "2px dashed #444",
        borderRadius: "12px",
        padding: "40px 80px",
        cursor: "pointer",
        marginBottom: "20px",
        textAlign: "center"
      }}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
        {preview ? (
          <img
            src={preview}
            alt="preview"
            style={{ maxWidth: "400px", borderRadius: "8px" }}
          />
        ) : (
          <div>
            <p style={{ fontSize: "3rem" }}>📁</p>
            <p>Image upload karo</p>
            <p style={{ color: "#666", fontSize: "0.8rem" }}>
              Click to browse
            </p>
          </div>
        )}
      </label>

      
      <button
        onClick={handleDetect}
        disabled={!file || loading}
        style={{
          background: file ? "#e53e3e" : "#333",
          color: "white",
          border: "none",
          padding: "14px 40px",
          borderRadius: "8px",
          fontSize: "1rem",
          cursor: file ? "pointer" : "not-allowed",
          marginBottom: "30px"
        }}
      >
        {loading ? "⏳ Detecting..." : "🔍 Detect Accident"}
      </button>

      
      {result && (
        <div style={{
          background: result.accident_detected ? "#2d0000" : "#002d00",
          border: `2px solid ${result.accident_detected ? "#e53e3e" : "#38a169"}`,
          borderRadius: "12px",
          padding: "30px 60px",
          textAlign: "center"
        }}>
          <p style={{ fontSize: "3rem" }}>
            {result.accident_detected ? "🚨" : "✅"}
          </p>
          <h2 style={{ color: result.accident_detected ? "#e53e3e" : "#38a169" }}>
            {result.accident_detected ? "ACCIDENT DETECTED!" : "No Accident"}
          </h2>
          <p>🚗 Vehicles Detected: {result.vehicles_detected}</p>
        </div>
      )}

    </div>
  )
}

export default App