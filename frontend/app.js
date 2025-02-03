document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const fileInput = document.getElementById('eeg-file');
  const resultDiv = document.getElementById('result');
  const predictionText = document.getElementById('prediction');
  const channelsList = document.getElementById('channels-list');

  form.addEventListener('submit', async (event) => {
      event.preventDefault();
      
      const file = fileInput.files[0];
      if (!file || !file.name.endsWith('.npy')) {
          alert('Please upload a valid .npy file.');
          return;
      }

      const formData = new FormData();
      formData.append('file', file);
      
      try {
          // Show loading indicator while awaiting response
          predictionText.innerHTML = "Processing...";
          channelsList.innerHTML = '';
          resultDiv.style.display = 'none';

          const response = await fetch("https://your-backend.onrender.com/predict/", {
              method: 'POST',
              body: formData
          });

          if (!response.ok) {
              throw new Error('Failed to fetch prediction');
          }

          const result = await response.json();

          // Display prediction result
          predictionText.innerHTML = `Predicted Class: ${result.predicted_class}`;
          resultDiv.style.display = 'block';

          // Display top channels
          result.top_channels.forEach(([channel, importance]) => {
              const li = document.createElement('li');
              li.innerHTML = `<span>${channel}:</span> ${importance.toFixed(4)}`;
              channelsList.appendChild(li);
          });
      } catch (error) {
          console.error('Error:', error);
          alert('An error occurred. Please try again later.');
      }
  });
});


  










"http://localhost:8000/predict/"