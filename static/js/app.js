// API Configuration
const API_BASE_URL = window.location.origin;

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await makePrediction();
});

// Make prediction
async function makePrediction() {
    const formData = new FormData(document.getElementById('predictionForm'));
    const data = {};
    
    // Convert form data to JSON
    for (let [key, value] of formData.entries()) {
        // Convert numeric fields
        if (['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
             'NumCreditLines', 'LoanTerm'].includes(key)) {
            data[key] = parseInt(value);
        } else if (['InterestRate', 'DTIRatio'].includes(key)) {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }
    
    // Show loading overlay
    document.getElementById('loadingOverlay').style.display = 'flex';
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction: ' + error.message);
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';
    }
}

// Display results
function displayResults(result) {
    const resultsCard = document.getElementById('resultsCard');
    const resultsContent = document.getElementById('resultsContent');
    
    const probability = (result.default_probability * 100).toFixed(2);
    const riskClass = `risk-${result.risk_level.toLowerCase()}`;
    const predictionClass = result.prediction === 'Default' ? 'prediction-default' : 'prediction-no-default';
    const progressColor = result.risk_level === 'Low' ? '#10b981' : 
                         result.risk_level === 'Medium' ? '#f59e0b' : '#ef4444';
    
    resultsContent.innerHTML = `
        <div class="result-summary">
            <div class="result-item">
                <div class="result-item-label">Prediction</div>
                <div class="result-item-value ${predictionClass}">
                    ${result.prediction}
                </div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Risk Level</div>
                <div class="result-item-value">
                    <span class="risk-badge ${riskClass}">${result.risk_level}</span>
                </div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Default Probability</div>
                <div class="result-item-value">${probability}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${probability}%; background-color: ${progressColor};">
                        ${probability}%
                    </div>
                </div>
            </div>
            <div class="result-item">
                <div class="result-item-label">Confidence Score</div>
                <div class="result-item-value">${(result.confidence * 100).toFixed(2)}%</div>
            </div>
        </div>
        
        <div class="details-grid">
            <div class="detail-item">
                <div class="detail-label">Classification Threshold</div>
                <div class="detail-value">${(result.threshold * 100).toFixed(2)}%</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Timestamp</div>
                <div class="detail-value">${new Date(result.timestamp).toLocaleString()}</div>
            </div>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 8px; border-left: 4px solid ${progressColor};">
            <h3 style="margin-bottom: 10px; color: #1e293b;">Interpretation</h3>
            <p style="color: #64748b; line-height: 1.6;">
                ${getInterpretation(result)}
            </p>
        </div>
    `;
    
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Get interpretation text
function getInterpretation(result) {
    const probability = (result.default_probability * 100).toFixed(2);
    
    if (result.risk_level === 'Low') {
        return `This loan application shows a <strong>low risk</strong> of default with a ${probability}% probability. 
                The applicant demonstrates strong creditworthiness and financial stability. 
                The confidence score of ${(result.confidence * 100).toFixed(2)}% indicates high certainty in this prediction.`;
    } else if (result.risk_level === 'Medium') {
        return `This loan application presents a <strong>moderate risk</strong> with a ${probability}% default probability. 
                Further review and additional risk mitigation strategies may be warranted. 
                Consider requiring additional documentation or a co-signer to reduce risk.`;
    } else {
        return `This loan application indicates a <strong>high risk</strong> of default at ${probability}% probability. 
                Careful consideration is recommended before approval. 
                Consider declining or requiring significant risk mitigation measures such as higher interest rates, 
                shorter loan terms, or substantial collateral.`;
    }
}

// Fill sample data
function fillSampleData() {
    const sampleData = {
        Age: 35,
        Income: 75000,
        LoanAmount: 25000,
        CreditScore: 720,
        MonthsEmployed: 48,
        NumCreditLines: 5,
        InterestRate: 5.5,
        LoanTerm: 60,
        DTIRatio: 0.35,
        Education: "Bachelor's",
        EmploymentType: "Full-time",
        MaritalStatus: "Married",
        HasMortgage: "Yes",
        HasDependents: "Yes",
        LoanPurpose: "Home",
        HasCoSigner: "No"
    };
    
    Object.keys(sampleData).forEach(key => {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = sampleData[key];
        }
    });
}

// Show about modal
function showAbout() {
    document.getElementById('aboutModal').style.display = 'flex';
}

// Close about modal
function closeAbout() {
    document.getElementById('aboutModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('aboutModal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
}

// Add input validation hints
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
        const min = this.getAttribute('min');
        const max = this.getAttribute('max');
        const value = parseFloat(this.value);
        
        if (min && value < parseFloat(min)) {
            this.style.borderColor = '#ef4444';
        } else if (max && value > parseFloat(max)) {
            this.style.borderColor = '#ef4444';
        } else {
            this.style.borderColor = '#e2e8f0';
        }
    });
});

// Add form validation feedback
document.getElementById('predictionForm').addEventListener('invalid', function(e) {
    e.preventDefault();
    const firstInvalid = this.querySelector(':invalid');
    if (firstInvalid) {
        firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
        firstInvalid.focus();
    }
}, true);

// Success message on form reset
document.getElementById('predictionForm').addEventListener('reset', function() {
    document.getElementById('resultsCard').style.display = 'none';
});

// Check API health on page load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const health = await response.json();
        console.log('API Health:', health);
        
        if (health.status !== 'healthy') {
            console.warn('API is not healthy:', health);
        }
    } catch (error) {
        console.error('Failed to check API health:', error);
        alert('Warning: Could not connect to the prediction API. Please ensure the server is running.');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    checkAPIHealth();
    console.log('Credit Risk Prediction App Loaded');
});
