const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const fs = require("fs").promises;
const path = require("path");

const app = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(express.static("public"));

let vectorizerParams;
let modelCoef;
let modelIntercept;
let classNames;
let diseaseDB;

async function loadAssets() {
  try {
    const vecData = await fs.readFile("./vectorizer_params.json", "utf8");
    vectorizerParams = JSON.parse(vecData);
    console.log("✅ Vectorizer params loaded");

    const modelData = await fs.readFile("./model_coef.json", "utf8");
    const model = JSON.parse(modelData);
    modelCoef = model.coef;
    modelIntercept = model.intercept;
    classNames = model.classes;
    console.log("✅ Model coefficients loaded");

    const dbData = await fs.readFile("./disease_db.json", "utf8");
    diseaseDB = JSON.parse(dbData);
    console.log("✅ Disease DB loaded");
  } catch (err) {
    console.error("❌ Failed to load assets:", err);
    process.exit(1);
  }
}

loadAssets().then(() => {
  module.exports = app;

  // For local development only
  if (require.main === module) {
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`🚀 Server running locally at http://localhost:${PORT}`);
    });
  }
});

function vectorizeText(text) {
  const words = text.toLowerCase().split(/\W+/);
  const vocab = vectorizerParams.vocabulary;
  const idf = vectorizerParams.idf;
  const maxFeatures = 5000;

  const termFreq = {};
  words.forEach((word) => {
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
  const exps = logits.map((l) => Math.exp(l - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sumExps);
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

  const probs = softmax(logits);
  return probs;
}

app.post("/predict", async (req, res) => {
  const { text } = req.body;
  if (!text || typeof text !== "string") {
    return res.status(400).json({ error: "Invalid input. Provide text." });
  }

  try {
    const probs = predict(text);

    const indices = probs
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3);

    const topMatches = indices.map((idx) => ({
      disease: classNames[idx.i],
      confidence: idx.p,
    }));

    const result = topMatches.map((match) => ({
      ...match,
      details: diseaseDB[match.disease] || null,
    }));

    res.json({ predictions: result });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Prediction failed" });
  }
});
