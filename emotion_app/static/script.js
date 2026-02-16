// ----------------------
// Text Emotion Analysis
// ----------------------
async function analyzeText() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) {
        alert("Please type something!");
        return;
    }

    const resultDiv = document.getElementById('text-result');
    const chartDiv = document.getElementById('text-chart');
    
    resultDiv.innerHTML = '<p>🔄 Analyzing...</p>';
    chartDiv.innerHTML = '';

    try {
        const res = await fetch('/analyze_text', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text})
        });
        
        const data = await res.json();
        
        if (data.error) {
            resultDiv.innerHTML = `<p style="color:#ff4444">❌ Error: ${data.error}</p>`;
            return;
        }
        
        displayResult('text-result', data);
        displayChart('text-chart', data.all_emotions);
    } catch (error) {
        resultDiv.innerHTML = `<p style="color:#ff4444">❌ Error: ${error.message}</p>`;
    }
}

// ----------------------
// Speech Emotion Analysis
// ----------------------
let recorder, audioStream;

async function startRecording() {
    const statusDiv = document.getElementById('recording-status');
    const resultDiv = document.getElementById('speech-result');
    const chartDiv = document.getElementById('speech-chart');
    
    resultDiv.innerHTML = '';
    chartDiv.innerHTML = '';
    
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({audio: true});
        recorder = new MediaRecorder(audioStream);
        let audioChunks = [];

        recorder.ondataavailable = e => audioChunks.push(e.data);
        
        recorder.onstop = async () => {
            statusDiv.innerHTML = '🔄 Processing audio...';
            
            const audioBlob = new Blob(audioChunks, {type: 'audio/webm'});
            const formData = new FormData();
            formData.append('audio', audioBlob, 'record.webm');
            
            // Get selected language from dropdown
            const selectedLanguage = document.getElementById('languageSelect').value;
            formData.append('language', selectedLanguage);

            try {
                const res = await fetch('/analyze_speech', {method:'POST', body: formData});
                const data = await res.json();
                
                if (data.error) {
                    resultDiv.innerHTML = `<p style="color:#ff4444">❌ Error: ${data.error}</p>`;
                    statusDiv.innerHTML = '';
                } else {
                    displayResult('speech-result', data);
                    displayChart('speech-chart', data.all_emotions);
                    statusDiv.innerHTML = '✅ Analysis complete!';
                    setTimeout(() => statusDiv.innerHTML = '', 3000);
                }
            } catch (error) {
                resultDiv.innerHTML = `<p style="color:#ff4444">❌ Error: ${error.message}</p>`;
                statusDiv.innerHTML = '';
            }

            audioStream.getTracks().forEach(track => track.stop());
        };

        recorder.start();
        statusDiv.innerHTML = '🔴 Recording... Speak now!';
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
        
    } catch (error) {
        statusDiv.innerHTML = `<p style="color:#ff4444">❌ Microphone access denied</p>`;
    }
}

function stopRecording() {
    if (recorder && recorder.state !== 'inactive') {
        recorder.stop();
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        document.getElementById('recording-status').innerHTML = '⏹ Recording stopped. Processing...';
    }
}

// ----------------------
// Display Results
// ----------------------
function displayResult(id, data) {
    const container = document.getElementById(id);
    
    let html = `
        <p><strong>Detected Emotion:</strong> ${data.emoji} ${data.emotion.toUpperCase()} 
        <span style="color:#4caf50">(${data.confidence}% confidence)</span></p>
    `;
    
    if (data.type === 'speech' && data.transcription) {
        html += `
            <div class="transcription">
                <p><strong>🌍 Detected Language:</strong> ${data.detected_language}</p>
                <p><strong>📝 English Translation:</strong> "${data.transcription}"</p>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

// ----------------------
// Display Horizontal Bar Chart
// ----------------------
function displayChart(id, emotions) {
    const container = document.getElementById(id);
    container.innerHTML = '<h3 style="margin-bottom: 20px; color: #ff9800;">📊 Emotion Distribution</h3>';
    
    emotions.forEach(emotion => {
        const barDiv = document.createElement('div');
        barDiv.className = 'chart-bar';
        
        barDiv.innerHTML = `
            <div class="chart-label">
                ${emotion.emoji} ${emotion.emotion}
            </div>
            <div class="chart-bar-bg">
                <div class="chart-bar-fill" 
                     style="width: ${emotion.confidence}%; background-color: ${emotion.color};">
                    ${emotion.confidence}%
                </div>
            </div>
        `;
        
        container.appendChild(barDiv);
    });
}