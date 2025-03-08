import { useState } from "react";

const EEGPrediction = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [channels, setChannels] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!file || !file.name.endsWith(".npy")) {
      alert("Please upload a valid .npy file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    
    setLoading(true);
    setPrediction(null);
    setChannels([]);

    try {
      const response = await fetch("https://eeg-detection.onrender.com/predict/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const result = await response.json();
      setPrediction(result.predicted_class);
      setChannels(result.top_channels);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>EEG Prediction</h1>
      <p>Upload your EEG data file (.npy format) and get predictions.</p>

      <form onSubmit={handleSubmit} encType="multipart/form-data">
        <div>
          <label htmlFor="eeg-file">Upload EEG Data File</label>
          <input type="file" id="eeg-file" accept=".npy" onChange={handleFileChange} required />
        </div>
        <button type="submit" disabled={loading}>Predict</button>
        {loading && <div className="loader"></div>}
      </form>

      {prediction && (
        <div id="result">
          <h3>Prediction Result:</h3>
          <p><strong>Predicted Class:</strong> {prediction}</p>
          <h3>Channel Importance</h3>
          <ul>
            {channels.map(([channel, importance], index) => (
              <li key={index}>
                <span>{channel}:</span> {importance.toFixed(4)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default EEGPrediction;