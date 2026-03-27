/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  regularizations,
  getKeyFromValue,
  Problem
} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';

interface NetworkExportConfig {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  activation: string;
  learningRate: number;
  regularization: string;
  regularizationRate: number;
  batchSize: number;
  problemType: string;
  featureNames: string[];
  trainTestSplit: number;
  epochs: number;
  weights?: number[][][];
  biases?: number[][];
}

const TOUR_STEPS = [
  {
    // Step 1: Data
    title: "Get Your Data Ready",
    content: `
      <h2>Step 1: Choose Your Data</h2>
      <p>Neural networks learn from examples. You need data with:</p>
      <ul>
        <li><strong>Features:</strong> Input values (like age, income, hours worked)</li>
        <li><strong>Labels:</strong> What you want to predict (like "will buy" or "price")</li>
      </ul>
      <div class="tour-option-cards">
        <div class="tour-option-card" data-action="upload-csv">
          <i class="material-icons">upload_file</i>
          <h3>Upload CSV</h3>
          <p>Use your own data file</p>
        </div>
        <div class="tour-option-card" data-action="use-builtin">
          <i class="material-icons">dataset</i>
          <h3>Built-in Data</h3>
          <p>Start with sample datasets</p>
        </div>
      </div>
    `
  },
  {
    // Step 2: Network Design
    title: "Design Your Network",
    content: `
      <h2>Step 2: Build Your Neural Network</h2>
      <p>A neural network has layers of "neurons" that process information:</p>
      <ul>
        <li><strong>Input Layer:</strong> Takes your data features</li>
        <li><strong>Hidden Layers:</strong> Processes and learns patterns</li>
        <li><strong>Output Layer:</strong> Makes predictions</li>
      </ul>
      <div class="tour-recommendation">
        <p><strong>Recommended for beginners:</strong> Start with 2 hidden layers, 4 neurons each. You can always adjust later!</p>
      </div>
      <p style="margin-top: 20px;">More neurons = more learning power, but slower training.</p>
    `
  },
  {
    // Step 3: Training
    title: "Train Your Network",
    content: `
      <h2>Step 3: Train and Watch It Learn</h2>
      <p>Training is when the network learns patterns from your data.</p>
      <p>Click the <strong>Play ▶</strong> button to start training. You'll see:</p>
      <ul>
        <li><strong>Lines changing color:</strong> The network is adjusting its "weights"</li>
        <li><strong>Loss decreasing:</strong> The network is getting better at predictions</li>
        <li><strong>Neurons lighting up:</strong> Different neurons activate for different patterns</li>
      </ul>
      <p>Training happens in "epochs" - each epoch means the network has seen all your data once.</p>
    `
  },
  {
    // Step 4: Results
    title: "Analyze Results",
    content: `
      <h2>Step 4: Check How Well It Learned</h2>
      <p>After training, you can see:</p>
      <ul>
        <li><strong>Accuracy:</strong> Percentage of correct predictions (for classification)</li>
        <li><strong>Loss:</strong> How wrong the predictions are (lower is better)</li>
        <li><strong>Prediction Table:</strong> Compare predicted vs actual values</li>
      </ul>
      <div class="tour-recommendation">
        <p><strong>Want to improve?</strong> Try training longer, adding more neurons, or adjusting the learning rate in the toolbar.</p>
      </div>
      <p style="margin-top: 24px; font-size: 18px; font-weight: 600; color: #6b5ce7;">🎉 You're ready to explore!</p>
    `
  }
];

let mainWidth;

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

let currentTourStep = 0;
let tourActive = false;

function initializeLandingAndTour() {
  // Check if user has disabled landing screen
  let dontShowLanding = localStorage.getItem('neuralnet_skip_landing');

  if (dontShowLanding === 'true') {
    // Hide landing, show main interface
    d3.select("#landing-screen").style("display", "none");
  } else {
    // Show landing screen
    d3.select("#landing-screen").style("display", "flex");
  }

  // Landing screen buttons
  d3.select("#start-tour-btn").on("click", () => {
    d3.select("#landing-screen").style("display", "none");
    startTour();
  });

  d3.select("#skip-to-playground-btn").on("click", () => {
    d3.select("#landing-screen").style("display", "none");
  });

  d3.select("#dont-show-landing").on("change", function() {
    if (this.checked) {
      localStorage.setItem('neuralnet_skip_landing', 'true');
    } else {
      localStorage.removeItem('neuralnet_skip_landing');
    }
  });

  // Tour navigation
  d3.select("#tour-skip-btn").on("click", () => {
    endTour();
  });

  d3.select("#tour-next-btn").on("click", () => {
    nextTourStep();
  });

  d3.select("#tour-back-btn").on("click", () => {
    previousTourStep();
  });
}

function startTour() {
  tourActive = true;
  currentTourStep = 0;
  d3.select("#tour-overlay").style("display", "block");
  updateTourStep();
}

function endTour() {
  tourActive = false;
  d3.select("#tour-overlay").style("display", "none");
}

function nextTourStep() {
  if (currentTourStep < TOUR_STEPS.length - 1) {
    currentTourStep++;
    updateTourStep();
  } else {
    // Tour complete
    endTour();
  }
}

function previousTourStep() {
  if (currentTourStep > 0) {
    currentTourStep--;
    updateTourStep();
  }
}

function updateTourStep() {
  let step = TOUR_STEPS[currentTourStep];

  // Update step indicator
  d3.select("#tour-current-step").text(currentTourStep + 1);

  // Update content
  d3.select("#tour-step-content").html(step.content);

  // Update navigation buttons
  if (currentTourStep === 0) {
    d3.select("#tour-back-btn").style("display", "none");
  } else {
    d3.select("#tour-back-btn").style("display", "inline-flex");
  }

  if (currentTourStep === TOUR_STEPS.length - 1) {
    d3.select("#tour-next-btn").html('<i class="material-icons">check</i> Finish Tour');
  } else {
    d3.select("#tour-next-btn").html('Next <i class="material-icons">arrow_forward</i>');
  }

  // Add click handlers for tour option cards (Step 1)
  d3.selectAll(".tour-option-card").on("click", function() {
    let action = this.getAttribute("data-action");

    if (action === "upload-csv") {
      // Trigger CSV upload
      endTour();
      document.getElementById("csv-upload-input").click();
    } else if (action === "use-builtin") {
      // Close tour and user can select built-in dataset
      endTour();
    }
  });
}

const RECT_SIZE = 45;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

interface ParsedCSV {
  columnNames: string[];
  rows: number[][];
  numRows: number;
  numColumns: number;
}

interface DatasetMapping {
  featureColumns: string[];
  labelColumn: string;
  data: DataPoint[];
  problemType: Problem;
}

interface DataPoint {
  features: number[];
  label: number;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

// Track available and active input features
const ALL_INPUT_FEATURES = ["x", "y", "xSquared", "ySquared", "xTimesY", "sinX", "sinY"];

let HIDABLE_CONTROLS = [
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
];

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (iter === 0) {
        simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      oneStep();
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart: AppendingLineChart | null = null;

function initLineChart() {
  if (lineChart) {
    return;
  }
  let container = d3.select("#linechart-bottom");
  if (container.empty()) {
    return;
  }
  lineChart = new AppendingLineChart(container, ["#777", "black"]);
}

let uploadedDataset: DatasetMapping | null = null;
let isUsingUploadedData = false;

let connectionEditMode = false;
let disabledConnections: {[id: string]: boolean} = {};

function clearDisabledConnections() {
  disabledConnections = {};
}

// Initialize input features - only x and y active by default
function initializeInputFeatures() {
  ALL_INPUT_FEATURES.forEach(feature => {
    if (feature === "x" || feature === "y") {
      state[feature] = true;
    } else {
      state[feature] = false;
    }
  });
}

function setupInputFeatureControls() {
  let lockedMessage = "Cannot change input features when using uploaded CSV. Upload a new CSV to select different features.";
  d3.select("#add-input-button").on("click", () => {
    if (isUsingUploadedData) {
      alert(lockedMessage);
      return;
    }
    addNextInputFeature();
  });

  d3.select("#remove-input-button").on("click", () => {
    if (isUsingUploadedData) {
      alert(lockedMessage);
      return;
    }
    removeLastInputFeature();
  });

  // Existing controls (if present)
  d3.select("#add-input-feature").on("click", () => {
    if (isUsingUploadedData) {
      alert(lockedMessage);
      return;
    }
    addNextInputFeature();
  });

  d3.select("#remove-input-feature").on("click", () => {
    if (isUsingUploadedData) {
      alert(lockedMessage);
      return;
    }
    removeLastInputFeature();
  });
  
  updateInputFeatureButtons();
}

function addNextInputFeature() {
  // Find next inactive feature
  for (let feature of ALL_INPUT_FEATURES) {
    if (!state[feature]) {
      state[feature] = true;
      parametersChanged = true;
      reset();
      updateInputFeatureButtons();
      return;
    }
  }
  // All features already active
  console.log("All input features are already active");
}

function removeLastInputFeature() {
  // Don't allow removing if only 1 feature left
  let activeCount = ALL_INPUT_FEATURES.filter(f => state[f]).length;
  if (activeCount <= 1) {
    console.log("Cannot remove - at least 1 input feature required");
    return;
  }
  
  // Find last active feature and remove it
  for (let i = ALL_INPUT_FEATURES.length - 1; i >= 0; i--) {
    let feature = ALL_INPUT_FEATURES[i];
    if (state[feature]) {
      state[feature] = false;
      parametersChanged = true;
      reset();
      updateInputFeatureButtons();
      return;
    }
  }
}

function updateInputFeatureButtons() {
  let activeCount = ALL_INPUT_FEATURES.filter(f => state[f]).length;
  let allCount = ALL_INPUT_FEATURES.length;
  
  // Disable add button if all features active
  d3.select("#add-input-feature")
    .attr("disabled", activeCount >= allCount ? "disabled" : null);
  
  // Disable remove button if only 1 feature left
  d3.select("#remove-input-feature")
    .attr("disabled", activeCount <= 1 ? "disabled" : null);
}

function setupConnectionEditControls() {
  let toggle = d3.select("#toggle-edit-connections");
  if (!toggle.empty()) {
    toggle.on("click", () => {
      connectionEditMode = !connectionEditMode;
      
      if (connectionEditMode) {
        enterEditMode();
      } else {
        exitEditMode();
      }
    });
  }
  
  let resetBtn = d3.select("#reset-connections");
  if (!resetBtn.empty()) {
    resetBtn.on("click", () => {
      if (confirm("Reset all connections to fully connected?")) {
        clearDisabledConnections();
        nn.resetAllConnections(network);
        updateUI();
        parametersChanged = true;

        if (connectionEditMode) {
          bindLinkClickHandlers();
        }
      }
    });
  }
}

function bindLinkClickHandlers() {
  d3.selectAll("#network .link-hover").on("click", function() {
    let linkId = d3.select(this).attr("data-link-id");
    if (linkId) {
      handleLinkClick(linkId);
    }
    (d3.event as Event).stopPropagation();
  });
}

function enterEditMode() {
  d3.select("body").classed("connection-edit-mode", true);
  
  let toggle = d3.select("#toggle-edit-connections");
  if (!toggle.empty()) {
    toggle
      .classed("active", true)
      .html('<i class="material-icons">done</i><span>Done Editing</span>');
  }
  
  let resetBtn = d3.select("#reset-connections");
  if (!resetBtn.empty()) {
    resetBtn.style("display", "block");
  }
  
  bindLinkClickHandlers();
}

function exitEditMode() {
  d3.select("body").classed("connection-edit-mode", false);
  
  let toggle = d3.select("#toggle-edit-connections");
  if (!toggle.empty()) {
    toggle
      .classed("active", false)
      .html('<i class="material-icons">edit</i><span>Edit Connections</span>');
  }
  
  let resetBtn = d3.select("#reset-connections");
  if (!resetBtn.empty()) {
    resetBtn.style("display", "none");
  }
  
  // Remove link click handlers
  d3.selectAll("#network .link-hover").on("click", null);
}

function applyDisabledConnectionsToNetwork() {
  if (!network) {
    return;
  }
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let layer = network[layerIdx];
    for (let i = 0; i < layer.length; i++) {
      let node = layer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let id = `${link.source.id}-${link.dest.id}`;
        link.isDisabled = !!disabledConnections[id];
      }
    }
  }
}

function handleLinkClick(linkId: string) {
  let parts = linkId.split("-");
  if (parts.length < 2) {
    return;
  }
  let sourceId = parts[0];
  let destId = parts.slice(1).join("-");
  
  if (!nn.canToggleConnection(network, sourceId, destId)) {
    alert("Cannot disable this connection - each neuron must have at least one active input connection.");
    return;
  }
  
  let isNowDisabled = nn.toggleConnection(network, sourceId, destId);
  
  if (isNowDisabled) {
    disabledConnections[linkId] = true;
  } else {
    delete disabledConnections[linkId];
  }
  
  updateUI();
  parametersChanged = true;
}

function parseCSVLine(line: string): string[] {
  let result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    let ch = line[i];
    if (ch === "\"") {
      if (inQuotes && i + 1 < line.length && line[i + 1] === "\"") {
        current += "\"";
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += ch;
    }
  }
  result.push(current.trim());
  return result;
}

function parseCSV(csvText: string): ParsedCSV {
  let lines = csvText.split(/\r?\n/).filter(line => line.trim() !== "");
  if (lines.length < 2) throw new Error("CSV must have header + data rows");
  
  let headers = parseCSVLine(lines[0]);
  let allRows: string[][] = [];
  
  for (let i = 1; i < lines.length; i++) {
    let values = parseCSVLine(lines[i]);
    if (values.length < headers.length) {
      while (values.length < headers.length) {
        values.push("");
      }
    }
    allRows.push(values);
  }
  
  // Identify numeric columns using an 80% threshold.
  let numericColumnIndices: number[] = [];
  let numericColumnNames: string[] = [];
  
  for (let colIdx = 0; colIdx < headers.length; colIdx++) {
    let numericCount = 0;
    for (let i = 0; i < allRows.length; i++) {
      let raw = (allRows[i][colIdx] || "").trim();
      let val = parseFloat(raw);
      if (!isNaN(val) && isFinite(val)) {
        numericCount++;
      }
    }
    let numericRatio = numericCount / allRows.length;
    if (numericRatio >= 0.8) {
      numericColumnIndices.push(colIdx);
      numericColumnNames.push(headers[colIdx]);
    }
  }
  
  if (numericColumnNames.length < 2) {
    let foundCols = numericColumnNames.length ? numericColumnNames.join(", ") : "none";
    throw new Error(`Need at least 2 numeric columns (1 label + 1 feature). Found numeric columns: ${foundCols}`);
  }
  
  let rows: number[][] = [];
  for (let i = 0; i < allRows.length; i++) {
    let row = allRows[i];
    let numericRow: number[] = [];
    for (let j = 0; j < numericColumnIndices.length; j++) {
      let colIdx = numericColumnIndices[j];
      let val = parseFloat((row[colIdx] || "").trim());
      numericRow.push(isNaN(val) ? 0 : val);
    }
    rows.push(numericRow);
  }
  
  return {
    columnNames: numericColumnNames,
    rows: rows,
    numRows: rows.length,
    numColumns: numericColumnNames.length
  };
}

function scaleValue(value: number, oldMin: number, oldMax: number, newMin: number, newMax: number): number {
  if (oldMax === oldMin) {
    return (newMin + newMax) / 2;
  }
  return ((value - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin;
}

function normalizeMultiDimensionalData(data: DataPoint[]): DataPoint[] {
  let numFeatures = data[0].features.length;
  
  // Find min/max for each feature
  let mins: number[] = [];
  let maxs: number[] = [];
  for (let i = 0; i < numFeatures; i++) {
    mins.push(Infinity);
    maxs.push(-Infinity);
  }
  
  for (let point of data) {
    for (let i = 0; i < numFeatures; i++) {
      mins[i] = Math.min(mins[i], point.features[i]);
      maxs[i] = Math.max(maxs[i], point.features[i]);
    }
  }
  
  // Normalize each feature to [-6, 6]
  let normalized: DataPoint[] = [];
  for (let point of data) {
    let normalizedFeatures: number[] = [];
    for (let i = 0; i < numFeatures; i++) {
      let val = scaleValue(point.features[i], mins[i], maxs[i], -6, 6);
      normalizedFeatures.push(val);
    }
    normalized.push({ features: normalizedFeatures, label: point.label });
  }
  
  // Normalize labels
  let labels = data.map(d => d.label);
  let labelMin = Math.min(...labels);
  let labelMax = Math.max(...labels);
  
  for (let i = 0; i < normalized.length; i++) {
    normalized[i].label = scaleValue(data[i].label, labelMin, labelMax, -1, 1);
  }
  
  return normalized;
}

function detectProblemType(labels: number[]): Problem {
  let seen: {[key: string]: boolean} = {};
  let uniqueCount = 0;
  for (let i = 0; i < labels.length; i++) {
    let key = String(labels[i]);
    if (!seen[key]) {
      seen[key] = true;
      uniqueCount++;
      if (uniqueCount > 10) {
        return Problem.REGRESSION;
      }
    }
  }
  return Problem.CLASSIFICATION;
}

function discretizeLabels(data: DataPoint[]): DataPoint[] {
  let seen: {[key: string]: boolean} = {};
  let uniqueLabels: number[] = [];
  for (let i = 0; i < data.length; i++) {
    let label = data[i].label;
    let key = String(label);
    if (!seen[key]) {
      seen[key] = true;
      uniqueLabels.push(label);
    }
  }
  uniqueLabels.sort((a, b) => a - b);
  
  if (uniqueLabels.length === 2) {
    // Binary: map to -1 and 1
    let [low, high] = uniqueLabels;
    return data.map(d => ({
      features: d.features,
      label: d.label === low ? -1 : 1
    }));
  } else {
    // Multi-class: threshold at 0
    return data.map(d => ({
      features: d.features,
      label: d.label >= 0 ? 1 : -1
    }));
  }
}

function showFeatureSelector(parsed: ParsedCSV) {
  let selector = d3.select("#feature-selector");
  selector.style("display", "block");
  selector.html("");
  
  selector.append("h4").text("Select Features and Label");
  
  // Label selection
  selector.append("label").text("Target/Label Column:");
  let labelSelect = selector.append("select").attr("id", "label-select");
  parsed.columnNames.forEach(name => {
    labelSelect.append("option").attr("value", name).text(name);
  });
  
  selector.append("br");
  selector.append("br");
  
  // Feature selection (checkboxes, max 10)
  selector.append("label").text("Select Features (1-10):");
  selector.append("div").attr("class", "feature-checkbox-container");
  
  let featureContainer = selector.select(".feature-checkbox-container");
  
  parsed.columnNames.forEach(name => {
    let div = featureContainer.append("div").attr("class", "feature-checkbox-item");
    let checkbox = div.append("input")
      .attr("type", "checkbox")
      .attr("id", "feature-" + name)
      .attr("value", name)
      .property("checked", true);
    
    div.append("label")
      .attr("for", "feature-" + name)
      .text(name);
    
    // Limit to 10 features
    checkbox.on("change", function() {
      let checkedCount = featureContainer.selectAll("input:checked").size();
      if (checkedCount > 10) {
        this.checked = false;
        alert("Maximum 10 features allowed");
      }
    });
  });
  
  selector.append("br");
  
  selector.append("button")
    .attr("class", "basic-button")
    .text("Use This Data")
    .on("click", () => applyCSVData(parsed));
  
  d3.select("#csv-upload-status").text(`Loaded ${parsed.numRows} rows, ${parsed.numColumns} numeric columns`);
}

function applyCSVData(parsed: ParsedCSV) {
  let labelColumn = (d3.select("#label-select").node() as HTMLSelectElement).value;
  
  let selectedFeatures: string[] = [];
  d3.selectAll(".feature-checkbox-container input:checked").each(function() {
    let name = (this as HTMLInputElement).value;
    if (name !== labelColumn) {
      selectedFeatures.push(name);
    }
  });
  
  if (selectedFeatures.length === 0) {
    alert("Please select at least 1 feature");
    return;
  }
  
  if (selectedFeatures.length > 10) {
    alert("Maximum 10 features allowed");
    return;
  }
  
  // Build dataset
  let labelIdx = parsed.columnNames.indexOf(labelColumn);
  let featureIndices = selectedFeatures.map(f => parsed.columnNames.indexOf(f));
  
  let dataPoints: DataPoint[] = [];
  for (let row of parsed.rows) {
    let features = featureIndices.map(idx => row[idx]);
    let label = row[labelIdx];
    dataPoints.push({ features, label });
  }
  
  // Normalize features
  let normalized = normalizeMultiDimensionalData(dataPoints);
  
  // Detect problem type
  let labels = normalized.map(d => d.label);
  let problemType = detectProblemType(labels);
  
  if (problemType === Problem.CLASSIFICATION) {
    normalized = discretizeLabels(normalized);
  }
  
  uploadedDataset = {
    featureColumns: selectedFeatures,
    labelColumn: labelColumn,
    data: normalized,
    problemType: problemType
  };
  
  state.problem = problemType;
  isUsingUploadedData = true;
  
  d3.select("#problem").property("value", getKeyFromValue(problems, problemType));
  d3.select("#feature-selector").style("display", "none");
  d3.select("#csv-upload-status").html(`<strong>Dataset loaded:</strong> ${selectedFeatures.length} features, ${normalized.length} samples`);
  d3.select("#csv-upload-status-display").html(`
    <p><strong>Dataset Loaded</strong></p>
    <p>${selectedFeatures.length} features, ${normalized.length} samples</p>
    <p>Type: ${problemType === Problem.CLASSIFICATION ? "Classification" : "Regression"}</p>
  `);
  d3.select("#view-predictions-btn").style("display", "block");
  d3.select("#export-code-btn").style("display", "block");
  
  reset();
}


function generateDataFromUpload() {
  if (!uploadedDataset) return;
  
  shuffle(uploadedDataset.data);
  
  let splitIndex = Math.floor(uploadedDataset.data.length * state.percTrainData / 100);
  let trainPoints = uploadedDataset.data.slice(0, splitIndex);
  let testPoints = uploadedDataset.data.slice(splitIndex);
  
  // Convert to old Example2D format for compatibility (use first 2 features as x, y for display only)
  trainData = trainPoints.map(p => ({
    x: p.features[0] || 0,
    y: p.features[1] || 0,
    label: p.label
  }));
  
  testData = testPoints.map(p => ({
    x: p.features[0] || 0,
    y: p.features[1] || 0,
    label: p.label
  }));
}

function setupCSVUpload() {
  d3.select("#csv-upload-input").on("change", function() {
    let fileList = (this as HTMLInputElement).files;
    if (!fileList || fileList.length === 0) {
      return;
    }
    let file = fileList[0];

    let reader = new FileReader();
    reader.onload = function() {
      try {
        let csvText = reader.result as string;
        let parsed = parseCSV(csvText);
        showFeatureSelector(parsed);
      } catch (error) {
        let message = (error instanceof Error) ? error.message : String(error);
        alert("Error: " + message);
      }
    };
    reader.readAsText(file);
  });
}

function makeGUI() {
  // Initialize landing and tour FIRST
  initializeLandingAndTour();

  d3.select("#restart-tour-btn").on("click", () => {
    startTour();
  });

  d3.select("#toggle-article-btn").on("click", function() {
    let article = d3.select("#article-text");
    let btn = d3.select(this);

    if (article.style("display") === "none") {
      article.style("display", "block");
      btn.classed("expanded", true);
      btn.html('<i class="material-icons">expand_less</i> Hide Details');
    } else {
      article.style("display", "none");
      btn.classed("expanded", false);
      btn.html('<i class="material-icons">expand_more</i> Learn More About Neural Networks');
    }
  });

  // Initialize input features
  initializeInputFeatures();
  
  // Setup input feature controls
  setupInputFeatureControls();

  // Setup connection edit controls
  setupConnectionEditControls();

  // Setup CSV upload controls
  setupCSVUpload();
  initLineChart();
  
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
    d3.select("#play-pause-button-toolbar").classed("playing", isPlaying);
    if (isPlaying) {
      d3.select("#play-pause-button-toolbar")
        .html('<i class="material-icons">pause</i> Pause');
    } else {
      d3.select("#play-pause-button-toolbar")
        .html('<i class="material-icons">play_arrow</i> Play');
    }
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#data-regen-button").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  let percTrain = d3.select("#percTrainData");
  if (!percTrain.empty()) {
    percTrain.on("input", function() {
      state.percTrainData = this.value;
      d3.select("label[for='percTrainData'] .value").text(this.value);
      generateData();
      parametersChanged = true;
      reset();
    });
    percTrain.property("value", state.percTrainData);
    d3.select("label[for='percTrainData'] .value").text(state.percTrainData);
  }

  let noise = d3.select("#noise");
  if (!noise.empty()) {
    noise.on("input", function() {
      state.noise = this.value;
      d3.select("label[for='noise'] .value").text(this.value);
      generateData();
      parametersChanged = true;
      reset();
    });
    let currentMax = parseInt(noise.property("max"));
    if (state.noise > currentMax) {
      if (state.noise <= 80) {
        noise.property("max", state.noise);
      } else {
        state.noise = 50;
      }
    } else if (state.noise < 0) {
      state.noise = 0;
    }
    noise.property("value", state.noise);
    d3.select("label[for='noise'] .value").text(state.noise);
  }

  let batchSize = d3.select("#batchSize");
  if (!batchSize.empty()) {
    batchSize.on("input", function() {
      state.batchSize = this.value;
      d3.select("label[for='batchSize'] .value").text(this.value);
      parametersChanged = true;
      reset();
    });
    batchSize.property("value", state.batchSize);
    d3.select("label[for='batchSize'] .value").text(state.batchSize);
  }

  let activationDropdown = d3.select("#activations");
  if (!activationDropdown.empty()) {
    activationDropdown.on("change", function() {
      state.activation = activations[this.value];
      parametersChanged = true;
      reset();
    });
    activationDropdown.property("value",
        getKeyFromValue(activations, state.activation));
  }

  let learningRate = d3.select("#learningRate");
  if (!learningRate.empty()) {
    learningRate.on("change", function() {
      state.learningRate = +this.value;
      state.serialize();
      userHasInteracted();
      parametersChanged = true;
    });
    learningRate.property("value", state.learningRate);
  }

  let regularDropdown = d3.select("#regularizations");
  if (!regularDropdown.empty()) {
    regularDropdown.on("change", function() {
      state.regularization = regularizations[this.value];
      parametersChanged = true;
      reset();
    });
    regularDropdown.property("value",
        getKeyFromValue(regularizations, state.regularization));
  }

  let regularRate = d3.select("#regularRate");
  if (!regularRate.empty()) {
    regularRate.on("change", function() {
      state.regularizationRate = +this.value;
      parametersChanged = true;
      reset();
    });
    regularRate.property("value", state.regularizationRate);
  }

  let problem = d3.select("#problem");
  if (!problem.empty()) {
    problem.on("change", function() {
      state.problem = problems[this.value];
      generateData();
      parametersChanged = true;
      reset();
    });
    problem.property("value", getKeyFromValue(problems, state.problem));
  }

  // Wire up toolbar controls (in addition to existing controls)
  d3.select("#reset-button-toolbar").on("click", () => {
    reset();
    userHasInteracted();
  });

  d3.select("#play-pause-button-toolbar").on("click", function () {
    userHasInteracted();
    player.playOrPause();
  });

  d3.select("#next-step-button-toolbar").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
    oneStep();
  });

  d3.select("#learningRate-toolbar").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
    let oldControl = d3.select("#learningRate").node();
    if (oldControl) {
      d3.select("#learningRate").property("value", this.value);
    }
  });

  d3.select("#activations-toolbar").on("change", function() {
    state.activation = activations[this.value];
    parametersChanged = true;
    reset();
    let oldControl = d3.select("#activations").node();
    if (oldControl) {
      d3.select("#activations").property("value", this.value);
    }
  });

  d3.select("#regularizations-toolbar").on("change", function() {
    state.regularization = regularizations[this.value];
    parametersChanged = true;
    reset();
    let oldControl = d3.select("#regularizations").node();
    if (oldControl) {
      d3.select("#regularizations").property("value", this.value);
    }
  });

  d3.select("#regularRate-toolbar").on("change", function() {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
    let oldControl = d3.select("#regularRate").node();
    if (oldControl) {
      d3.select("#regularRate").property("value", this.value);
    }
  });

  d3.select("#problem-toolbar").on("change", function() {
    state.problem = problems[this.value];
    generateData();
    parametersChanged = true;
    reset();
    let oldControl = d3.select("#problem").node();
    if (oldControl) {
      d3.select("#problem").property("value", this.value);
    }
  });

  d3.select("#add-layers-toolbar").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      alert("Maximum 6 hidden layers allowed");
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers-toolbar").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      alert("Must have at least 1 hidden layer");
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  d3.select("#toggle-edit-connections-toolbar").on("click", () => {
    let existingBtn = d3.select("#toggle-edit-connections").node() as HTMLButtonElement | null;
    if (existingBtn) {
      existingBtn.click();
    } else {
      connectionEditMode = !connectionEditMode;
      if (connectionEditMode) {
        enterEditMode();
      } else {
        exitEditMode();
      }
    }
  });

  // Sliders in bottom panel
  d3.select("#batchSize-slider").on("input", function() {
    state.batchSize = +this.value;
    d3.select("#batch-display").text(this.value);
    let oldControl = d3.select("#batchSize").node();
    if (oldControl) {
      d3.select("#batchSize").property("value", this.value);
    }
    parametersChanged = true;
    reset();
  });

  d3.select("#noise-slider").on("input", function() {
    state.noise = +this.value;
    d3.select("#noise-display").text(this.value);
    let oldControl = d3.select("#noise").node();
    if (oldControl) {
      d3.select("#noise").property("value", this.value);
    }
    generateData();
    parametersChanged = true;
    reset();
  });

  d3.select("#percTrainData-slider").on("input", function() {
    state.percTrainData = +this.value;
    d3.select("#split-display").text(this.value);
    let oldControl = d3.select("#percTrainData").node();
    if (oldControl) {
      d3.select("#percTrainData").property("value", this.value);
    }
    generateData();
    parametersChanged = true;
    reset();
  });

  d3.select("#data-regen-button-bottom").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  // View predictions modal
  d3.select("#view-predictions-btn").on("click", () => {
    d3.select("#prediction-modal").style("display", "flex");
  });

  d3.select("#close-modal-btn").on("click", () => {
    d3.select("#prediction-modal").style("display", "none");
  });

  // Export code button - show when data is uploaded
  d3.select("#export-code-btn").on("click", () => {
    // Open modal
    d3.select("#code-export-modal").style("display", "flex");
    
    // Generate and display code
    updateCodePreview();
  });

  // Close modal button
  d3.select("#close-code-modal").on("click", () => {
    d3.select("#code-export-modal").style("display", "none");
  });

  // Close modal when clicking backdrop
  d3.select("#code-export-modal .modal-backdrop").on("click", () => {
    d3.select("#code-export-modal").style("display", "none");
  });

  // Update code when checkboxes change
  d3.select("#include-comments").on("change", () => {
    updateCodePreview();
  });

  d3.select("#include-weights").on("change", () => {
    updateCodePreview();
  });

  // Copy to clipboard button
  d3.select("#copy-code-btn").on("click", () => {
    copyCodeToClipboard();
  });

  // Initialize slider displays
  d3.select("#batch-display").text(state.batchSize);
  d3.select("#noise-display").text(state.noise);
  d3.select("#split-display").text(state.percTrainData);

  d3.select("#batchSize-slider").property("value", state.batchSize);
  d3.select("#noise-slider").property("value", state.noise);
  d3.select("#percTrainData-slider").property("value", state.percTrainData);

  // Listen for css-responsive changes and redraw the svg network.

  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part-redesign")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }

  // Initialize scroll effects
  initializeScrollEffects();
}

function initializeScrollEffects() {
  let toolbar = d3.select("#top-toolbar");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 50) {
      toolbar.classed("scrolled", true);
    } else {
      toolbar.classed("scrolled", false);
    }
  });
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateNeuronColors(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    // Color main neuron rectangle based on output value.
    d3.select(`#node${node.id} > rect:first-child`)
      .style("fill", colorScale(node.output || 0));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .classed("disabled", !!link.isDisabled)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let isUploadedInput = isUsingUploadedData && uploadedDataset != null;
  let inputIsActive = isUploadedInput ? true : !!state[nodeId];
  
  // Skip drawing if this is an inactive input feature
  if (isInput && !inputIsActive) {
    return; // Don't draw anything for inactive features
  }
  
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });
    
  let activeOrNotClass = inputIsActive ? "active" : "inactive";
  
  if (isInput) {
    let label: string;
    if (isUsingUploadedData) {
      // Show actual feature names from CSV.
      label = nodeId;
    } else {
      // Original behavior.
      let inputFeature = INPUTS[nodeId];
      label = (inputFeature != null && inputFeature.label != null) ?
          inputFeature.label : nodeId;
    }
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas - only for active input features
  if (!isInput || inputIsActive) {
    let div = d3.select("#network").insert("div", ":first-child")
      .attr({
        "id": `canvas-${nodeId}`,
        "class": "canvas"
      })
      .style({
        position: "absolute",
        left: `${x + 3}px`,
        top: `${y + 3}px`
      });
      
    if (isInput) {
      div.style("cursor", "default"); // Remove click functionality for input nodes
    }
    
    if (isInput) {
      div.classed(activeOrNotClass, true);
    }
    
    div.datum({id: nodeId});
  }
}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  let padding = 3;
  let containerNode = d3.select("#network-container").node() as HTMLDivElement;
  let width = containerNode ? containerNode.clientWidth - padding * 2 : 800;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
      .domain(d3.range(1, numLayers - 1))
      .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 40);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately - only active features
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = isUsingUploadedData && uploadedDataset != null ?
      uploadedDataset.featureColumns : Object.keys(INPUTS);
  
  // Filter to only active features
  let activeNodeIds = (isUsingUploadedData && uploadedDataset != null) ?
      nodeIds : nodeIds.filter(nodeId => state[nodeId]);
  
  let maxY = nodeIndexScale(activeNodeIds.length);
  activeNodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length).node() as any;
        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2;
  node2coord[node.id] = {cx, cy};
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
  }
  // Adjust the height of the svg.
  svg.attr("height", maxY);
  svg.attr("viewBox", `0 0 ${width} ${maxY}`);
  svg.attr("preserveAspectRatio", "xMidYMid meet");

  // Legacy layout height adjustment no longer applies in redesigned UI.
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement | null;
  if (!node) {
    return 0;
  }
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x}px`); // Centered with transform in CSS

  let i = layerIdx - 1;
  
  // Add neuron control buttons
  let buttonContainer = div.append("div")
    .attr("class", "neuron-controls");
  
  buttonContainer.append("button")
    .attr("class", "neuron-control-button add-button")
    .on("click", () => {
      let numNeurons = state.networkShape[i];
      if (numNeurons >= 8) {
        return;
      }
      state.networkShape[i]++;
      parametersChanged = true;
      reset();
    })
    .html('<i class="material-icons">add</i><span>Add Neuron</span>');

  buttonContainer.append("button")
    .attr("class", "neuron-control-button remove-button")
    .on("click", () => {
      let numNeurons = state.networkShape[i];
      if (numNeurons <= 1) {
        return;
      }
      state.networkShape[i]--;
      parametersChanged = true;
      reset();
    })
    .html('<i class="material-icons">remove</i><span>Remove Neuron</span>');

  // Neuron count display
  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div")
    .attr("class", "neuron-count")
    .text(state.networkShape[i] + " neuron" + suffix);
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .attr("data-link-id", `${input.source.id}-${input.dest.id}`)
    .on("mouseenter", function() {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function() {
      updateHoverCard(null);
    });
  return line;
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let uploadedPoint = (isUsingUploadedData && uploadedDataset != null) ?
        uploadedDataset.data[i % uploadedDataset.data.length] : null;
    let input = constructInput(dataPoints[i].x, dataPoints[i].y, uploadedPoint);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoints[i].label);
  }
  return loss / dataPoints.length;
}

function updatePredictionTable() {
  if (!isUsingUploadedData || uploadedDataset == null || testData.length === 0) {
    d3.select("#prediction-results").style("display", "none");
    return;
  }
  
  d3.select("#prediction-results").style("display", "block");
  
  let predictions: {predicted: number, actual: number, match: boolean}[] = [];
  let correct = 0;
  
  for (let i = 0; i < testData.length; i++) {
    let dataPoint = uploadedDataset.data[trainData.length + i];
    let input = constructInput(testData[i].x, testData[i].y, dataPoint);
    let output = nn.forwardProp(network, input);
    let actual = testData[i].label;
    
    let predictedClass = output >= 0 ? 1 : -1;
    let actualClass = actual >= 0 ? 1 : -1;
    let match = predictedClass === actualClass;
    
    if (match) correct++;
    
    predictions.push({
      predicted: output,
      actual: actual,
      match: match
    });
  }
  
  let accuracy = (correct / testData.length * 100).toFixed(1);
  d3.select("#accuracy-display").html(`<strong>Accuracy:</strong> ${accuracy}% (${correct}/${testData.length} correct)`);
  
  // Show first 10 predictions
  let tbody = d3.select("#prediction-tbody");
  tbody.html("");
  
  let displayCount = Math.min(10, predictions.length);
  for (let i = 0; i < displayCount; i++) {
    let row = tbody.append("tr");
    row.append("td").text(i + 1);
    row.append("td").text(predictions[i].predicted.toFixed(3));
    row.append("td").text(predictions[i].actual.toFixed(3));
    row.append("td")
      .attr("class", predictions[i].match ? "match-correct" : "match-incorrect")
      .text(predictions[i].match ? "✓" : "✗");
  }
  
  if (predictions.length > 10) {
    d3.select("#show-all-predictions")
      .style("display", "block")
      .text(`Show All ${predictions.length} Predictions`);
  } else {
    d3.select("#show-all-predictions").style("display", "none");
  }
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  updateNeuronColors(network);

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#loss-train-display").text(humanReadable(lossTrain));
  d3.select("#loss-test-display").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  d3.select("#iter-number-toolbar").text(addCommas(zeroPad(iter)));
  if (lineChart) {
    lineChart.addDataPoint([lossTrain, lossTest]);
  }
  
  // Update prediction table
  updatePredictionTable();

  // Update layer display
  d3.select("#num-layers-display").text(state.numHiddenLayers);

  // Show accuracy if classification
  if (isUsingUploadedData && uploadedDataset.problemType === Problem.CLASSIFICATION) {
    d3.select("#accuracy-metric").style("display", "block");
  }

  if (connectionEditMode) {
    bindLinkClickHandlers();
  }
}

function constructInputIds(): string[] {
  if (isUsingUploadedData && uploadedDataset != null) {
    return uploadedDataset.featureColumns.slice();
  }
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number, dataPoint?: DataPoint): number[] {
  if (isUsingUploadedData && dataPoint) {
    // Use actual features from uploaded data
    return dataPoint.features;
  }
  
  // Original behavior for built-in datasets
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

function oneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    let dataPoint = (isUsingUploadedData && uploadedDataset != null) ?
        uploadedDataset.data[i % uploadedDataset.data.length] : null;
    let input = constructInput(point.x, point.y, dataPoint);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset(onStartup=false) {
  if (lineChart) {
    lineChart.reset();
  }
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);
  
  if (isUsingUploadedData) {
    generateDataFromUpload();
  }

  // Make a simple network.
  iter = 0;
  let numInputs = (isUsingUploadedData && uploadedDataset != null) ?
      uploadedDataset.featureColumns.length : constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let outputActivation = (state.problem === Problem.REGRESSION) ?
      nn.Activations.LINEAR : nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation,
      state.regularization, constructInputIds(), state.initZero);
  applyDisabledConnectionsToNetwork();
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  updateUI(true);
};

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
      .text(title.text());
      document.title = title.text();
    }
  });
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

function generateData(firstTime = false) {
  if (isUsingUploadedData) {
    generateDataFromUpload();
    return;
  }
  
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let numSamples = (state.problem === Problem.REGRESSION) ?
      NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  if (typeof ga !== 'undefined') {
    let page = 'index';
    if (state.tutorial != null && state.tutorial !== '') {
      page = `/v/tutorials/${state.tutorial}`;
    }
    ga('set', 'page', page);
    ga('send', 'pageview', {'sessionControl': 'start'});
  }
}

function simulationStarted() {
  if (typeof ga !== 'undefined') {
    ga('send', {
      hitType: 'event',
      eventCategory: 'Starting Simulation',
      eventAction: parametersChanged ? 'changed' : 'unchanged',
      eventLabel: state.tutorial == null ? '' : state.tutorial
    });
  }
  parametersChanged = false;
}

function generateKerasCode(includeComments: boolean, includeWeights: boolean): string {
  // Collect network configuration
  let config: NetworkExportConfig = {
    inputSize: isUsingUploadedData ? uploadedDataset.featureColumns.length : constructInput(0, 0).length,
    hiddenLayers: state.networkShape.slice(0, state.numHiddenLayers),
    outputSize: 1,
    activation: getKeyFromValue(activations, state.activation),
    learningRate: state.learningRate,
    regularization: state.regularization ? getKeyFromValue(regularizations, state.regularization) : 'none',
    regularizationRate: state.regularizationRate,
    batchSize: state.batchSize,
    problemType: state.problem === Problem.CLASSIFICATION ? 'classification' : 'regression',
    featureNames: isUsingUploadedData ? uploadedDataset.featureColumns : Object.keys(INPUTS).filter(k => state[k]),
    trainTestSplit: state.percTrainData / 100,
    epochs: iter
  };
  
  // Extract weights if requested
  if (includeWeights && network) {
    config.weights = extractWeights(network);
    config.biases = extractBiases(network);
  }
  
  return buildKerasCodeString(config, includeComments, includeWeights);
}

function extractWeights(network: nn.Node[][]): number[][][] {
  let weights: number[][][] = [];
  
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let layerWeights: number[][] = [];
    let layer = network[layerIdx];
    
    // For each neuron in this layer
    for (let nodeIdx = 0; nodeIdx < layer.length; nodeIdx++) {
      let node = layer[nodeIdx];
      let nodeWeights: number[] = [];
      
      // Get weights from all input links
      for (let link of node.inputLinks) {
        nodeWeights.push(link.weight);
      }
      
      layerWeights.push(nodeWeights);
    }
    
    weights.push(layerWeights);
  }
  
  return weights;
}

function extractBiases(network: nn.Node[][]): number[][] {
  let biases: number[][] = [];
  
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let layerBiases: number[] = [];
    let layer = network[layerIdx];
    
    for (let node of layer) {
      layerBiases.push(node.bias);
    }
    
    biases.push(layerBiases);
  }
  
  return biases;
}

function buildKerasCodeString(config: NetworkExportConfig, includeComments: boolean, includeWeights: boolean): string {
  let code = '';
  
  // Header
  if (includeComments) {
    code += `"""\n`;
    code += `Neural Network Training Code\n`;
    code += `Generated by NeuralNet++\n\n`;
    code += `Architecture: ${config.inputSize} inputs → [${config.hiddenLayers.join(', ')}] hidden → ${config.outputSize} output\n`;
    code += `Problem: ${config.problemType === 'classification' ? 'Classification' : 'Regression'}\n`;
    code += `Activation: ${config.activation}\n`;
    code += `Learning Rate: ${config.learningRate}\n`;
    if (config.regularization !== 'none') {
      code += `Regularization: ${config.regularization} (rate: ${config.regularizationRate})\n`;
    }
    code += `"""\n\n`;
  }
  
  // Imports
  code += `import numpy as np\n`;
  code += `from tensorflow import keras\n`;
  code += `from tensorflow.keras import layers\n`;
  if (config.regularization !== 'none') {
    code += `from tensorflow.keras import regularizers\n`;
  }
  code += `from sklearn.model_selection import train_test_split\n`;
  code += `from sklearn.preprocessing import StandardScaler\n\n`;
  
  // Section 1: Network Architecture
  if (includeComments) {
    code += `# ============================================\n`;
    code += `# 1. NETWORK ARCHITECTURE\n`;
    code += `# ============================================\n\n`;
  }
  
  code += `model = keras.Sequential([\n`;
  
  // Input + first hidden layer
  let firstLayer = config.hiddenLayers[0];
  let regStr = '';
  if (config.regularization === 'L1') {
    regStr = `, kernel_regularizer=regularizers.l1(${config.regularizationRate})`;
  } else if (config.regularization === 'L2') {
    regStr = `, kernel_regularizer=regularizers.l2(${config.regularizationRate})`;
  }
  
  if (includeComments) {
    code += `    # Input layer expects ${config.inputSize} features\n`;
  }
  code += `    layers.Dense(${firstLayer}, activation='${config.activation}', input_shape=(${config.inputSize},)${regStr}),\n`;
  
  // Remaining hidden layers
  for (let i = 1; i < config.hiddenLayers.length; i++) {
    if (includeComments) {
      code += `    # Hidden layer ${i + 1}\n`;
    }
    code += `    layers.Dense(${config.hiddenLayers[i]}, activation='${config.activation}'${regStr}),\n`;
  }
  
  // Output layer
  let outputActivation = config.problemType === 'regression' ? 'linear' : 'tanh';
  if (includeComments) {
    code += `    # Output layer\n`;
  }
  code += `    layers.Dense(${config.outputSize}, activation='${outputActivation}')\n`;
  code += `])\n\n`;
  
  // Section 2: Compile
  if (includeComments) {
    code += `# ============================================\n`;
    code += `# 2. COMPILE MODEL\n`;
    code += `# ============================================\n\n`;
  }
  
  code += `model.compile(\n`;
  code += `    optimizer=keras.optimizers.SGD(learning_rate=${config.learningRate}),\n`;
  code += `    loss='mse',\n`;
  code += `    metrics=['mae']\n`;
  code += `)\n\n`;
  
  // Section 3: Load weights (if included)
  if (includeWeights && config.weights && config.biases) {
    if (includeComments) {
      code += `# ============================================\n`;
      code += `# 3. LOAD TRAINED WEIGHTS\n`;
      code += `# ============================================\n\n`;
    }
    
    code += `# Trained weights from NeuralNet++\n`;
    code += `trained_weights = [\n`;
    
    for (let i = 0; i < config.weights.length; i++) {
      // Transpose weights for Keras format (inputs x outputs)
      let weightsMatrix = config.weights[i];
      let transposed = transposeMatrix(weightsMatrix);
      
      code += `    # Layer ${i + 1} weights\n`;
      code += `    np.array(${JSON.stringify(transposed)}),\n`;
      code += `    # Layer ${i + 1} biases\n`;
      code += `    np.array(${JSON.stringify(config.biases[i])}),\n`;
    }
    
    code += `]\n\n`;
    code += `model.set_weights(trained_weights)\n\n`;
  }
  
  // Section 4: Load Data
  let sectionNum = includeWeights ? 4 : 3;
  if (includeComments) {
    code += `# ============================================\n`;
    code += `# ${sectionNum}. LOAD YOUR DATA\n`;
    code += `# ============================================\n\n`;
    code += `# Replace this with your actual data\n`;
    code += `# Expected format:\n`;
    code += `#   X shape: (n_samples, ${config.inputSize}) - your features\n`;
    code += `#   y shape: (n_samples,) - your labels\n\n`;
  }
  
  code += `X = np.array([\n`;
  code += `    # Your data here (${config.featureNames.join(', ')})\n`;
  code += `    # Example row: [value1, value2, ...]\n`;
  code += `])\n\n`;
  code += `y = np.array([\n`;
  code += `    # Your labels here\n`;
  code += `])\n\n`;
  
  // Section 5: Preprocess
  sectionNum++;
  if (includeComments) {
    code += `# ============================================\n`;
    code += `# ${sectionNum}. PREPROCESS DATA\n`;
    code += `# ============================================\n\n`;
    code += `# Normalize features (same as NeuralNet++ does)\n`;
  }
  
  code += `scaler = StandardScaler()\n`;
  code += `X_scaled = scaler.fit_transform(X)\n\n`;
  
  if (includeComments) {
    code += `# Split into train/test\n`;
  }
  code += `X_train, X_test, y_train, y_test = train_test_split(\n`;
  code += `    X_scaled, y,\n`;
  code += `    test_size=${1 - config.trainTestSplit},\n`;
  code += `    random_state=42\n`;
  code += `)\n\n`;
  
  // Section 6: Train
  sectionNum++;
  if (includeComments) {
    code += `# ============================================\n`;
    code += `# ${sectionNum}. TRAIN THE MODEL\n`;
    code += `# ============================================\n\n`;
  }
  
  if (!includeWeights) {
    code += `history = model.fit(\n`;
    code += `    X_train, y_train,\n`;
    code += `    epochs=${config.epochs > 0 ? config.epochs : 1000},\n`;
    code += `    batch_size=${config.batchSize},\n`;
    code += `    validation_data=(X_test, y_test),\n`;
    code += `    verbose=1\n`;
    code += `)\n\n`;
  } else {
    if (includeComments) {
      code += `# Model already has trained weights loaded\n`;
      code += `# To retrain from scratch, remove the weights section above\n\n`;
    }
  }
  
  // Section 7: Evaluate
  sectionNum++;
  if (includeComments) {
    code += `# ============================================\n`;
    code += `# ${sectionNum}. EVALUATE\n`;
    code += `# ============================================\n\n`;
  }
  
  code += `test_loss = model.evaluate(X_test, y_test, verbose=0)\n`;
  code += `print(f'Test Loss: {test_loss[0]:.4f}')\n\n`;
  
  if (config.problemType === 'classification') {
    if (includeComments) {
      code += `# Calculate accuracy for classification\n`;
    }
    code += `predictions = model.predict(X_test, verbose=0)\n`;
    code += `predicted_classes = (predictions >= 0).astype(int).flatten()\n`;
    code += `actual_classes = (y_test >= 0).astype(int)\n`;
    code += `accuracy = np.mean(predicted_classes == actual_classes)\n`;
    code += `print(f'Accuracy: {accuracy * 100:.2f}%')\n`;
  }
  
  return code;
}

function transposeMatrix(matrix: number[][]): number[][] {
  if (matrix.length === 0) return [];
  let rows = matrix.length;
  let cols = matrix[0].length;
  let transposed: number[][] = [];
  
  for (let j = 0; j < cols; j++) {
    transposed[j] = [];
    for (let i = 0; i < rows; i++) {
      transposed[j][i] = matrix[i][j];
    }
  }
  
  return transposed;
}

function updateCodePreview() {
  let includeComments = (d3.select("#include-comments").node() as HTMLInputElement).checked;
  let includeWeights = (d3.select("#include-weights").node() as HTMLInputElement).checked;
  
  let code = generateKerasCode(includeComments, includeWeights);
  d3.select("#generated-code-preview code").text(code);
}

function copyCodeToClipboard() {
  let code = d3.select("#generated-code-preview code").text();

  let clipboard = (navigator as any).clipboard;
  if (clipboard && clipboard.writeText) {
    clipboard.writeText(code).then(() => {
      // Show success message
      d3.select("#copy-success-message").style("display", "inline-block");
      
      // Hide after 3 seconds
      setTimeout(() => {
        d3.select("#copy-success-message").style("display", "none");
      }, 3000);
    }).catch(err => {
      alert("Failed to copy code. Please select and copy manually.");
      console.error("Copy failed:", err);
    });
    return;
  }

  // Fallback for older browsers / TS lib definitions
  try {
    let textarea = document.createElement("textarea");
    textarea.value = code;
    textarea.setAttribute("readonly", "true");
    textarea.style.position = "absolute";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);

    d3.select("#copy-success-message").style("display", "inline-block");
    setTimeout(() => {
      d3.select("#copy-success-message").style("display", "none");
    }, 3000);
  } catch (err) {
    alert("Failed to copy code. Please select and copy manually.");
    console.error("Copy failed:", err);
  }
}

initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
