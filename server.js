const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const fs = require('fs').promises;
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(express.static('public')); // serve frontend

// Global variables
let vectorizerParams;
let modelCoef;
let modelIntercept;
let classNames;
let diseaseDB;

// Load all required JSON files
async function loadAssets() {
    try {
        const vecData = await fs.readFile(path.join(__dirname, 'vectorizer_params.json'), 'utf8');
        vectorizerParams = JSON.parse(vecData);
        console.log('✅ Vectorizer params loaded');
    } catch (err) {
        console.error('❌ Failed to load vectorizer_params.json:', err.message);
        throw err;
    }

    try {
        const modelData = await fs.readFile(path.join(__dirname, 'model_coef.json'), 'utf8');
        const model = JSON.parse(modelData);
        modelCoef = model.coef;
        modelIntercept = model.intercept;
        classNames = model.classes;
        console.log('✅ Model coefficients loaded');
    } catch (err) {
        console.error('❌ Failed to load model_coef.json:', err.message);
        throw err;
    }

    try {
        const dbData = await fs.readFile(path.join(__dirname, 'disease_db.json'), 'utf8');
        diseaseDB = JSON.parse(dbData);
        console.log('✅ Disease DB loaded');
    } catch (err) {
        console.error('❌ Failed to load disease_db.json:', err.message);
        throw err;
    }
}

// Start loading and attach the promise
const assetsPromise = loadAssets();

// Middleware to wait for assets before processing requests
app.use(async (req, res, next) => {
    try {
        await assetsPromise;
        next();
    } catch (err) {
        res.status(500).json({ error: 'Server failed to load required data.' });
    }
});

// ----- Helper functions (copied from your original) -----
function vectorizeText(text) {
    const words = text.toLowerCase().split(/\W+/);
    const vocab = vectorizerParams.vocabulary;
    const idf = vectorizerParams.idf;
    const maxFeatures = 5000;

    const termFreq = {};
    words.forEach(word => {
        if (vocab.hasOwnProperty(word)) {
            const idx = vocab[word];
            termFreq[idx] = (termFreq[idx] || 0) + 1;
        }
    });

    const totalWords = words.length;
    const vector = new Array(maxFeatures).fill(0);

    for (let idx in termFreq) {
        const tf = termFreq[idx] / totalWords;
        const idfValue = idf[idx];
        vector[idx] = tf * idfValue;
    }

    let l2Norm = 0;
    for (let i = 0; i < maxFeatures; i++) {
        l2Norm += vector[i] * vector[i];
    }
    l2Norm = Math.sqrt(l2Norm);
    if (l2Norm > 0) {
        for (let i = 0; i < maxFeatures; i++) {
            vector[i] /= l2Norm;
        }
    }
    return vector;
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sumExps);
}

function predict(text) {
    const x = vectorizeText(text);
    const nClasses = modelCoef.length;
    const logits = [];
    for (let c = 0; c < nClasses; c++) {
        let logit = modelIntercept[c];
        const coefRow = modelCoef[c];
        for (let i = 0; i < x.length; i++) {
            if (x[i] !== 0) {
                logit += x[i] * coefRow[i];
            }
        }
        logits.push(logit);
    }
    return softmax(logits);
}
// -------------------------------------------------------

// Prediction endpoint
app.post('/predict', async (req, res) => {
    const { text } = req.body;
    if (!text || typeof text !== 'string') {
        return res.status(400).json({ error: 'Invalid input. Provide text.' });
    }

    try {
        const probs = predict(text);
        const indices = probs
            .map((p, i) => ({ p, i }))
            .sort((a, b) => b.p - a.p)
            .slice(0, 3);

        const topMatches = indices.map(idx => ({
            disease: classNames[idx.i],
            confidence: idx.p
        }));

        const result = topMatches.map(match => ({
            ...match,
            details: diseaseDB[match.disease] || null
        }));

        res.json({ predictions: result });
    } catch (err) {
        console.error('Prediction error:', err);
        res.status(500).json({ error: 'Prediction failed' });
    }
});

// Health check endpoint (optional)
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Export the app for Vercel
module.exports = app;

// For local development only
if (require.main === module) {
    const PORT = process.env.PORT || 3000;
    assetsPromise.then(() => {
        app.listen(PORT, () => {
            console.log(`🚀 Server running locally at http://localhost:${PORT}`);
        });
    }).catch(err => {
        console.error('Failed to load assets, cannot start server:', err);
        process.exit(1);
    });
}