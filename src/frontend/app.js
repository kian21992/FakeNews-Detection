

function hideResults() {
    document.getElementById('results-container').classList.add('hidden');
    
    // Reset circle animation
    const circle = document.getElementById('score-circle');
    circle.style.strokeDashoffset = 283;
}

function hideError() {
    document.getElementById('error-banner').classList.add('hidden');
}

function showError(msg) {
    document.getElementById('error-message').textContent = msg;
    document.getElementById('error-banner').classList.remove('hidden');
    hideResults();
}

function setLoading(isLoading) {
    const btn = document.getElementById('analyze-btn');
    const icon = document.getElementById('btn-icon');
    const loader = document.getElementById('btn-loader');
    const text = document.getElementById('btn-text');
    
    btn.disabled = isLoading;
    if (isLoading) {
        icon.classList.add('hidden');
        loader.classList.remove('hidden');
        text.textContent = 'Analyzing...';
    } else {
        icon.classList.remove('hidden');
        loader.classList.add('hidden');
        text.textContent = 'Analyze Claim';
    }
}

async function analyze() {
    hideError();
    hideResults();
    
    const endpoint = 'http://localhost:8000/analyze/text';
    const payload = {};
    
    const text = document.getElementById('input-text').value.trim();
    if (!text) {
        showError("Please enter some text to analyze.");
        return;
    }
    payload.text = text;
    
    setLoading(true);
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || "An error occurred during analysis.");
        }
        
        if (data.prediction.error) {
           throw new Error(data.prediction.error);
        }
        
        displayResults(data);
        
    } catch (err) {
        showError(err.message);
    } finally {
        setLoading(false);
    }
}

function displayResults(data) {
    const isReliable = data.prediction.is_reliable;
    const score = data.prediction.confidence_score; // 0 to 1
    const percentage = Math.round(score * 100);
    
    // Update Badge
    const badge = document.getElementById('result-badge');
    if (isReliable) {
        badge.textContent = "Credible";
        badge.className = "ml-4 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide bg-green-500/20 text-green-400 border border-green-500/30";
    } else {
        badge.textContent = "Potentially Misleading";
        badge.className = "ml-4 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wide bg-red-500/20 text-red-400 border border-red-500/30";
    }
    
    // Update Text Detail
    const classText = document.getElementById('classification-text');
    if (isReliable) {
        classText.textContent = "Likely Reliable";
        classText.className = "text-xl font-bold text-green-400";
    } else {
        classText.textContent = "High Risk of Misinformation";
        classText.className = "text-xl font-bold text-red-400";
    }
    
    document.getElementById('confidence-text').textContent = (percentage > 90 ? "High" : percentage > 70 ? "Medium" : "Low") + ` (${percentage}%)`;
    
    // Animation of the Circle
    // circumference = 2 * pi * r = 2 * 3.14159 * 45 = ~283
    const circle = document.getElementById('score-circle');
    const offset = 283 - (percentage / 100) * 283;
    
    // Set color based on reliability
    circle.classList.remove('stroke-green-500', 'stroke-red-500', 'stroke-yellow-500');
    if (isReliable) {
        circle.classList.add('stroke-green-500');
    } else {
        circle.classList.add('stroke-red-500');
    }
    
    // Add small delay to allow display:block to apply before animation
    setTimeout(() => {
        circle.style.strokeDashoffset = offset;
        document.getElementById('score-value').textContent = percentage + "%";
    }, 50);
    
    // Store variables globally for feedback submission
    window.lastAnalyzedText = data.text || "";
    window.lastModelScore = score;
    
    // Reset feedback UI
    document.getElementById('feedback-section').classList.remove('hidden');
    document.getElementById('btn-feedback-real').classList.remove('hidden');
    document.getElementById('btn-feedback-fake').classList.remove('hidden');
    document.getElementById('feedback-thanks').classList.add('hidden');

    // Update Breakdown
    if (data.prediction.breakdown) {
        const bd = data.prediction.breakdown;
        
        // Transformer Score %
        document.getElementById('bd-transformer').textContent = Math.round(bd.transformer_base_score * 100) + "%";
        
        // Subjectivity (0 is fully objective, 1 is fully subjective)
        const subjScore = Math.round(bd.subjectivity * 100);
        const subjEl = document.getElementById('bd-subjectivity');
        subjEl.textContent = subjScore + "%";
        
        if (subjScore > 60) {
            subjEl.className = "font-semibold text-red-400";
            subjEl.textContent += " (Penalty Applied)";
        } else if (subjScore < 40) {
            subjEl.className = "font-semibold text-green-400";
        } else {
            subjEl.className = "font-semibold text-yellow-500";
        }
        

    }

    // Entities
    const entitiesContainer = document.getElementById('entities-container');
    const noEntitiesMsg = document.getElementById('no-entities-msg');
    
    entitiesContainer.innerHTML = '';
    
    if (data.entities && data.entities.length > 0) {
        noEntitiesMsg.classList.add('hidden');
        data.entities.forEach(ent => {
             const span = document.createElement('span');
             span.textContent = `${ent.text} (${ent.label})`;
             span.title = `Found ${ent.count} time(s)`;
             span.className = "px-3 py-1 bg-gray-700 hover:bg-gray-600 text-gray-200 text-xs rounded-full border border-gray-600 transition-colors cursor-help";
             entitiesContainer.appendChild(span);
        });
    } else {
        noEntitiesMsg.classList.remove('hidden');
    }
    

    
    // Show results
    document.getElementById('results-container').classList.remove('hidden');
}

async function submitFeedback(isReliable) {
    if (!window.lastAnalyzedText) return;
    
    // Hide buttons, show loading or thanks
    document.getElementById('btn-feedback-real').classList.add('hidden');
    document.getElementById('btn-feedback-fake').classList.add('hidden');
    document.getElementById('feedback-thanks').classList.remove('hidden');
    
    try {
        await fetch('http://localhost:8000/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: window.lastAnalyzedText,
                is_reliable: isReliable,
                model_score: window.lastModelScore || 0.5
            })
        });
    } catch(err) {
        console.error("Error submitting feedback:", err);
    }
}

document.getElementById('analyze-btn').addEventListener('click', analyze);
document.getElementById('btn-feedback-real').addEventListener('click', () => submitFeedback(true));
document.getElementById('btn-feedback-fake').addEventListener('click', () => submitFeedback(false));
