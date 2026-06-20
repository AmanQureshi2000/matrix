"""
Matrix Pro - Web Application for Render Deployment
Run with: python app.py
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np
from scipy.linalg import lu, expm, logm, svd, qr, schur, hessenberg
from scipy.linalg import solve, lstsq, pinv
from scipy.spatial.distance import pdist, squareform
from scipy.stats import describe
import re
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Matrix Pro - Web Edition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            transition: all 0.3s ease;
        }

        body.dark-theme {
            background: #1e1e2e;
            color: #cdd6f4;
        }

        .app-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 15px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            margin-bottom: 15px;
            border-bottom: 2px solid #dce1e8;
        }

        body.dark-theme .header {
            border-bottom-color: #3d3d5c;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 700;
            color: #2c3e50;
        }

        body.dark-theme .header h1 {
            color: #cdd6f4;
        }

        .header-actions {
            display: flex;
            gap: 10px;
        }

        .btn {
            background: #ecf0f1;
            color: #2c3e50;
            border: 1px solid #dce1e8;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn:hover {
            background: #dde1e6;
            border-color: #b0b8c4;
        }

        .btn:active {
            background: #ccd1d9;
        }

        body.dark-theme .btn {
            background: #3d3d5c;
            color: #cdd6f4;
            border-color: #4d4d6c;
        }

        body.dark-theme .btn:hover {
            background: #4d4d6c;
            border-color: #5d5d7c;
        }

        .btn-small {
            padding: 4px 10px;
            font-size: 11px;
        }

        .btn-theme, .btn-help {
            background: transparent;
            border: none;
            font-size: 14px;
        }

        .btn-theme:hover, .btn-help:hover {
            background: #ecf0f1;
            border-radius: 6px;
        }

        body.dark-theme .btn-theme:hover,
        body.dark-theme .btn-help:hover {
            background: #3d3d5c;
        }

        .main-layout {
            display: grid;
            grid-template-columns: 2fr 2fr 3fr;
            gap: 15px;
        }

        .panel {
            background: #ffffff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        body.dark-theme .panel {
            background: #2d2d44;
        }

        .left-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .matrix-group {
            background: #f8f9fc;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #dce1e8;
        }

        body.dark-theme .matrix-group {
            background: #2d2d44;
            border-color: #3d3d5c;
        }

        .matrix-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            flex-wrap: wrap;
            gap: 8px;
        }

        .matrix-header h3 {
            font-size: 14px;
            color: #2c3e50;
        }

        body.dark-theme .matrix-header h3 {
            color: #cdd6f4;
        }

        .matrix-controls {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }

        .matrix-controls label {
            font-size: 12px;
            margin: 0 2px;
        }

        .matrix-controls input {
            width: 45px;
            padding: 2px 4px;
            border: 1px solid #dce1e8;
            border-radius: 4px;
            font-size: 12px;
            background: #ffffff;
            color: #2c3e50;
        }

        body.dark-theme .matrix-controls input {
            background: #2d2d44;
            color: #cdd6f4;
            border-color: #3d3d5c;
        }

        .matrix-group textarea {
            width: 100%;
            min-height: 100px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            padding: 8px;
            border: 1px solid #dce1e8;
            border-radius: 6px;
            background: #f8f9fc;
            color: #2c3e50;
            resize: vertical;
        }

        body.dark-theme .matrix-group textarea {
            background: #1e1e2e;
            color: #cdd6f4;
            border-color: #3d3d5c;
        }

        .quick-fill {
            display: flex;
            gap: 6px;
            margin-top: 6px;
            flex-wrap: wrap;
        }

        .action-bar {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .center-panel {
            display: flex;
            flex-direction: column;
        }

        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 10px;
            flex-wrap: wrap;
            border-bottom: 2px solid #dce1e8;
            padding-bottom: 4px;
        }

        body.dark-theme .tabs {
            border-bottom-color: #3d3d5c;
        }

        .tab-btn {
            background: #ecf0f1;
            border: none;
            padding: 6px 14px;
            border-radius: 6px 6px 0 0;
            cursor: pointer;
            font-weight: 500;
            font-size: 12px;
            color: #7f8c9e;
            transition: all 0.2s;
        }

        body.dark-theme .tab-btn {
            background: #3d3d5c;
            color: #a6adc8;
        }

        .tab-btn:hover {
            background: #dde1e6;
        }

        body.dark-theme .tab-btn:hover {
            background: #4d4d6c;
        }

        .tab-btn.active {
            background: #ffffff;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
        }

        body.dark-theme .tab-btn.active {
            background: #2d2d44;
            color: #cdd6f4;
            border-bottom-color: #7f8c9e;
        }

        .tab-content {
            flex: 1;
            min-height: 200px;
        }

        .tab-pane {
            display: none;
            flex-wrap: wrap;
            gap: 6px;
            padding: 8px 0;
        }

        .tab-pane.active {
            display: flex;
        }

        .tab-pane .btn {
            flex: 0 0 auto;
        }

        .right-panel {
            display: flex;
            flex-direction: column;
        }

        .result-group {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .result-group h3 {
            font-size: 14px;
            margin-bottom: 8px;
            color: #2c3e50;
        }

        body.dark-theme .result-group h3 {
            color: #cdd6f4;
        }

        .result-display {
            flex: 1;
            min-height: 300px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            padding: 10px;
            background: #f8f9fc;
            border: 1px solid #dce1e8;
            border-radius: 6px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }

        body.dark-theme .result-display {
            background: #1e1e2e;
            color: #cdd6f4;
            border-color: #3d3d5c;
        }

        .result-info {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            font-size: 12px;
            color: #5a6b7c;
        }

        body.dark-theme .result-info {
            color: #7f8c9e;
        }

        .result-actions {
            display: flex;
            gap: 8px;
            margin-top: 6px;
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }

        .modal-content {
            background: #ffffff;
            margin: 5% auto;
            padding: 20px;
            border-radius: 12px;
            width: 80%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }

        body.dark-theme .modal-content {
            background: #2d2d44;
            color: #cdd6f4;
        }

        .close {
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }

        .close:hover {
            color: #999;
        }

        .help-content h3 {
            margin-top: 15px;
            margin-bottom: 8px;
        }

        .help-content ul {
            margin-left: 20px;
            margin-bottom: 10px;
        }

        .help-content li {
            margin-bottom: 4px;
        }

        @media (max-width: 1024px) {
            .main-layout {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 600px) {
            .header h1 {
                font-size: 20px;
            }
            .matrix-header {
                flex-direction: column;
                align-items: stretch;
            }
            .matrix-controls {
                flex-wrap: wrap;
            }
            .tab-btn {
                font-size: 10px;
                padding: 4px 10px;
            }
        }

        .status-ok {
            color: #27ae60;
        }
        
        .status-error {
            color: #e74c3c;
        }
        
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="app-container">
        <header class="header">
            <h1>📊 Matrix Pro</h1>
            <div class="header-actions">
                <button id="themeToggle" class="btn btn-theme">🌓 Toggle Theme</button>
                <button id="helpBtn" class="btn btn-help">❓ Help</button>
            </div>
        </header>

        <div class="main-layout">
            <div class="panel left-panel">
                <div class="matrix-group">
                    <div class="matrix-header">
                        <h3>Matrix A</h3>
                        <div class="matrix-controls">
                            <label>Rows:</label>
                            <input type="number" id="rowsA" value="3" min="1" max="12">
                            <label>Cols:</label>
                            <input type="number" id="colsA" value="3" min="1" max="12">
                            <button id="resizeA" class="btn btn-small">📐 Resize</button>
                        </div>
                    </div>
                    <textarea id="matrixA" placeholder="Enter values (space/comma per row)"></textarea>
                    <div class="quick-fill">
                        <button class="btn btn-small" onclick="fillMatrix('A', 'identity')">Identity</button>
                        <button class="btn btn-small" onclick="fillMatrix('A', 'zeros')">Zeros</button>
                        <button class="btn btn-small" onclick="fillMatrix('A', 'ones')">Ones</button>
                        <button class="btn btn-small" onclick="fillMatrix('A', 'random')">Random</button>
                    </div>
                </div>

                <div class="matrix-group">
                    <div class="matrix-header">
                        <h3>Matrix B</h3>
                        <div class="matrix-controls">
                            <label>Rows:</label>
                            <input type="number" id="rowsB" value="3" min="1" max="12">
                            <label>Cols:</label>
                            <input type="number" id="colsB" value="3" min="1" max="12">
                            <button id="resizeB" class="btn btn-small">📐 Resize</button>
                        </div>
                    </div>
                    <textarea id="matrixB" placeholder="Enter values (space/comma per row)"></textarea>
                    <div class="quick-fill">
                        <button class="btn btn-small" onclick="fillMatrix('B', 'identity')">Identity</button>
                        <button class="btn btn-small" onclick="fillMatrix('B', 'zeros')">Zeros</button>
                        <button class="btn btn-small" onclick="fillMatrix('B', 'ones')">Ones</button>
                        <button class="btn btn-small" onclick="fillMatrix('B', 'random')">Random</button>
                    </div>
                </div>

                <div class="action-bar">
                    <button class="btn" onclick="swapMatrices()">🔄 Swap A↔B</button>
                    <button class="btn" onclick="clearAll()">🗑️ Clear All</button>
                </div>
            </div>

            <div class="panel center-panel">
                <div class="tabs">
                    <button class="tab-btn active" data-tab="basic">Basic</button>
                    <button class="tab-btn" data-tab="advanced">Advanced</button>
                    <button class="tab-btn" data-tab="properties">Properties</button>
                    <button class="tab-btn" data-tab="elementwise">Element-wise</button>
                    <button class="tab-btn" data-tab="linalg">Linear Algebra</button>
                </div>
                <div class="tab-content">
                    <div id="tab-basic" class="tab-pane active">
                        <button class="btn" onclick="operate('add')">➕ Addition</button>
                        <button class="btn" onclick="operate('subtract')">➖ Subtraction</button>
                        <button class="btn" onclick="operate('multiply')">✖️ Multiplication</button>
                        <button class="btn" onclick="operate('transpose')">🔀 Transpose</button>
                        <button class="btn" onclick="operate('inverse')">🔄 Inverse</button>
                        <button class="btn" onclick="operate('determinant')">🧮 Determinant</button>
                        <button class="btn" onclick="operate('rank')">📊 Rank</button>
                        <button class="btn" onclick="operate('trace')">🔢 Trace</button>
                    </div>
                    <div id="tab-advanced" class="tab-pane">
                        <button class="btn" onclick="operate('eigenvalues')">🔲 Eigenvalues</button>
                        <button class="btn" onclick="operate('eigenvectors')">🔲 Eigenvectors</button>
                        <button class="btn" onclick="operate('frobenius_norm')">📐 Frobenius Norm</button>
                        <button class="btn" onclick="operate('infinity_norm')">📐 Infinity Norm</button>
                        <button class="btn" onclick="operate('l1_norm')">📐 L1 Norm</button>
                        <button class="btn" onclick="operate('condition')">🔍 Condition #</button>
                        <button class="btn" onclick="promptPower()">⚡ Matrix Power</button>
                        <button class="btn" onclick="operate('lu')">🧩 LU Decomp</button>
                        <button class="btn" onclick="operate('qr')">🧩 QR Decomp</button>
                        <button class="btn" onclick="operate('svd')">🧩 SVD</button>
                        <button class="btn" onclick="operate('cholesky')">🔲 Cholesky</button>
                        <button class="btn" onclick="operate('matrix_exp')">📈 Exponential</button>
                        <button class="btn" onclick="operate('matrix_log')">📉 Logarithm</button>
                        <button class="btn" onclick="operate('schur')">🔲 Schur Decomp</button>
                        <button class="btn" onclick="operate('hessenberg')">🔲 Hessenberg</button>
                    </div>
                    <div id="tab-properties" class="tab-pane">
                        <button class="btn" onclick="operate('size')">📏 Size</button>
                        <button class="btn" onclick="operate('is_square')">🔍 Square?</button>
                        <button class="btn" onclick="operate('is_symmetric')">🔍 Symmetric?</button>
                        <button class="btn" onclick="operate('is_skew_symmetric')">🔍 Skew-Sym?</button>
                        <button class="btn" onclick="operate('is_diagonal')">🔍 Diagonal?</button>
                        <button class="btn" onclick="operate('is_identity')">🔍 Identity?</button>
                        <button class="btn" onclick="operate('is_orthogonal')">🔍 Orthogonal?</button>
                        <button class="btn" onclick="operate('is_invertible')">🔍 Invertible?</button>
                        <button class="btn" onclick="operate('is_positive_definite')">🔍 Positive Def?</button>
                        <button class="btn" onclick="operate('is_nilpotent')">🔍 Nilpotent?</button>
                        <button class="btn" onclick="operate('is_idempotent')">🔍 Idempotent?</button>
                        <button class="btn" onclick="operate('is_hermitian')">🔍 Hermitian?</button>
                        <button class="btn" onclick="operate('is_unitary')">🔍 Unitary?</button>
                        <button class="btn" onclick="operate('is_normal')">🔍 Normal?</button>
                    </div>
                    <div id="tab-elementwise" class="tab-pane">
                        <button class="btn" onclick="operate('elementwise_mul')">✖️ Multiply</button>
                        <button class="btn" onclick="operate('elementwise_div')">➗ Divide</button>
                        <button class="btn" onclick="operate('elementwise_power')">⚡ Power</button>
                        <button class="btn" onclick="operate('elementwise_abs')">📐 Abs</button>
                        <button class="btn" onclick="operate('elementwise_sqrt')">📐 Sqrt</button>
                        <button class="btn" onclick="operate('elementwise_log')">📐 Log</button>
                        <button class="btn" onclick="operate('elementwise_exp')">📐 Exp</button>
                        <button class="btn" onclick="operate('elementwise_sin')">📐 Sin</button>
                        <button class="btn" onclick="operate('elementwise_cos')">📐 Cos</button>
                        <button class="btn" onclick="operate('elementwise_tan')">📐 Tan</button>
                        <button class="btn" onclick="operate('elementwise_sigmoid')">📐 Sigmoid</button>
                    </div>
                    <div id="tab-linalg" class="tab-pane">
                        <button class="btn" onclick="promptSolve()">📈 Solve Ax=b</button>
                        <button class="btn" onclick="promptLeastSquares()">📉 Least Squares</button>
                        <button class="btn" onclick="operate('pseudo_inverse')">🔢 Pseudo-inverse</button>
                        <button class="btn" onclick="operate('stats')">📊 Matrix Stats</button>
                        <button class="btn" onclick="operate('gram')">🔲 Gram Matrix</button>
                        <button class="btn" onclick="operate('covariance')">🔲 Covariance</button>
                        <button class="btn" onclick="operate('correlation')">🔲 Correlation</button>
                        <button class="btn" onclick="operate('distance')">📏 Distance Matrix</button>
                    </div>
                </div>
            </div>

            <div class="panel right-panel">
                <div class="result-group">
                    <h3>📊 Result</h3>
                    <div class="result-display" id="resultDisplay">Ready</div>
                    <div class="result-info">
                        <span id="operationInfo">Ready</span>
                        <span id="statusInfo"></span>
                    </div>
                    <div class="result-actions">
                        <button class="btn" onclick="copyResult()">📋 Copy</button>
                        <button class="btn" onclick="undoLast()">↩️ Undo</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="helpModal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>📚 Matrix Pro – Help</h2>
            <div class="help-content">
                <h3>Basic Usage</h3>
                <ul>
                    <li>Enter matrix values in the text areas (space or comma separated, one row per line)</li>
                    <li>Use <strong>Resize</strong> to change matrix dimensions</li>
                    <li>Quick fill buttons: Identity, Zeros, Ones, Random</li>
                </ul>
                <h3>Tabs</h3>
                <ul>
                    <li><strong>Basic:</strong> Addition, Subtraction, Multiplication, Transpose, Inverse, Determinant, Rank, Trace</li>
                    <li><strong>Advanced:</strong> Eigenvalues, Eigenvectors, Norms, Condition #, Power, LU/QR/SVD/Cholesky/Schur/Hessenberg, Exp, Log</li>
                    <li><strong>Properties:</strong> Check matrix properties (square, symmetric, etc.)</li>
                    <li><strong>Element-wise:</strong> Element-wise operations</li>
                    <li><strong>Linear Algebra:</strong> Solve Ax=b, Least Squares, Pseudo-inverse, Stats, Gram, Covariance, Correlation, Distance</li>
                </ul>
                <h3>Keyboard Shortcuts</h3>
                <ul>
                    <li><strong>Ctrl+C:</strong> Copy result</li>
                    <li><strong>Ctrl+Z:</strong> Undo last operation</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        let history = [];
        let currentResult = null;

        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
                this.classList.add('active');
                document.getElementById('tab-' + this.dataset.tab).classList.add('active');
            });
        });

        let darkTheme = false;
        document.getElementById('themeToggle').addEventListener('click', function() {
            darkTheme = !darkTheme;
            document.body.classList.toggle('dark-theme');
        });

        const modal = document.getElementById('helpModal');
        document.getElementById('helpBtn').onclick = function() {
            modal.style.display = 'block';
        }
        document.querySelector('.close').onclick = function() {
            modal.style.display = 'none';
        }
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }

        document.getElementById('resizeA').addEventListener('click', function() {
            resizeMatrix('A');
        });
        document.getElementById('resizeB').addEventListener('click', function() {
            resizeMatrix('B');
        });

        function resizeMatrix(name) {
            const rows = parseInt(document.getElementById('rows' + name).value) || 3;
            const cols = parseInt(document.getElementById('cols' + name).value) || 3;
            const textarea = document.getElementById('matrix' + name);
            const existing = textarea.value.trim();
            if (existing) {
                const lines = existing.split('\\n');
                const newLines = [];
                for (let i = 0; i < rows; i++) {
                    if (i < lines.length) {
                        const vals = lines[i].trim().split(/[,\\s]+/).filter(v => v);
                        while (vals.length < cols) vals.push('0');
                        newLines.push(vals.slice(0, cols).join(' '));
                    } else {
                        newLines.push('0 '.repeat(cols).trim());
                    }
                }
                textarea.value = newLines.join('\\n');
            } else {
                textarea.value = '0 '.repeat(cols).trim() + '\\n'.repeat(rows - 1) + '0 '.repeat(cols).trim();
            }
        }

        async function fillMatrix(name, type) {
            const rows = parseInt(document.getElementById('rows' + name).value) || 3;
            const cols = parseInt(document.getElementById('cols' + name).value) || 3;
            
            try {
                const response = await fetch('/api/fill', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ type, rows, cols })
                });
                const data = await response.json();
                if (data.success) {
                    document.getElementById('matrix' + name).value = data.str;
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Request failed: ' + error);
            }
        }

        function getMatrixData(name) {
            const textarea = document.getElementById('matrix' + name);
            return textarea.value;
        }

        function swapMatrices() {
            const a = document.getElementById('matrixA').value;
            const b = document.getElementById('matrixB').value;
            document.getElementById('matrixA').value = b;
            document.getElementById('matrixB').value = a;
            
            const rowsA = document.getElementById('rowsA').value;
            const colsA = document.getElementById('colsA').value;
            const rowsB = document.getElementById('rowsB').value;
            const colsB = document.getElementById('colsB').value;
            
            document.getElementById('rowsA').value = rowsB;
            document.getElementById('colsA').value = colsB;
            document.getElementById('rowsB').value = rowsA;
            document.getElementById('colsB').value = colsA;
        }

        async function operate(operation) {
            const matrixA = getMatrixData('A');
            const matrixB = getMatrixData('B');
            
            if (!matrixA.trim()) {
                alert('Matrix A is empty');
                return;
            }

            const payload = { operation, matrix_a: matrixA };
            
            const needsB = ['add', 'subtract', 'multiply', 'elementwise_mul', 'elementwise_div'];
            if (needsB.includes(operation) && !matrixB.trim()) {
                alert('Matrix B is required for this operation');
                return;
            }
            
            if (matrixB.trim()) {
                payload.matrix_b = matrixB;
            }

            if (operation === 'power' || operation === 'elementwise_power') {
                const power = prompt('Enter power:');
                if (power === null) return;
                const p = parseFloat(power);
                if (isNaN(p)) {
                    alert('Please enter a valid number');
                    return;
                }
                payload.power = p;
            }

            if (operation === 'solve' || operation === 'least_squares') {
                const b = prompt('Enter vector b (space/comma separated):');
                if (b === null) return;
                if (!b.trim()) {
                    alert('Vector b is required');
                    return;
                }
                payload.b = b;
            }

            const display = document.getElementById('resultDisplay');
            display.textContent = 'Processing...';
            display.classList.add('loading');

            try {
                const response = await fetch('/api/operate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                
                display.classList.remove('loading');
                
                if (data.success) {
                    displayResult(data.result, operation, data.info);
                } else {
                    display.textContent = 'Error: ' + data.error;
                    document.getElementById('statusInfo').textContent = '❌ Error';
                    document.getElementById('statusInfo').className = 'status-error';
                }
            } catch (error) {
                display.classList.remove('loading');
                display.textContent = 'Request failed: ' + error;
                document.getElementById('statusInfo').textContent = '❌ Connection Error';
                document.getElementById('statusInfo').className = 'status-error';
            }
        }

        function displayResult(result, operation, info) {
            const display = document.getElementById('resultDisplay');
            display.textContent = result;
            currentResult = result;
            document.getElementById('operationInfo').textContent = `${operation} | ${info || ''}`;
            document.getElementById('statusInfo').textContent = '✅ Completed';
            document.getElementById('statusInfo').className = 'status-ok';
            history.push({ result, operation, info });
        }

        function copyResult() {
            if (currentResult) {
                navigator.clipboard.writeText(currentResult).then(() => {
                    document.getElementById('statusInfo').textContent = '📋 Copied!';
                    document.getElementById('statusInfo').className = 'status-ok';
                }).catch(() => {
                    const display = document.getElementById('resultDisplay');
                    const range = document.createRange();
                    range.selectNode(display);
                    window.getSelection().removeAllRanges();
                    window.getSelection().addRange(range);
                    document.execCommand('copy');
                    document.getElementById('statusInfo').textContent = '📋 Copied!';
                });
            }
        }

        function undoLast() {
            if (history.length > 1) {
                history.pop();
                const last = history[history.length - 1];
                document.getElementById('resultDisplay').textContent = last.result;
                currentResult = last.result;
                document.getElementById('operationInfo').textContent = `Undo: ${last.operation} | ${last.info || ''}`;
                document.getElementById('statusInfo').textContent = '↩️ Undo';
                document.getElementById('statusInfo').className = 'status-ok';
            } else if (history.length === 1) {
                history.pop();
                document.getElementById('resultDisplay').textContent = 'Ready';
                currentResult = null;
                document.getElementById('operationInfo').textContent = 'Ready';
                document.getElementById('statusInfo').textContent = '↩️ Undo';
            } else {
                document.getElementById('statusInfo').textContent = 'Nothing to undo';
                document.getElementById('statusInfo').className = '';
            }
        }

        function clearAll() {
            document.getElementById('matrixA').value = '';
            document.getElementById('matrixB').value = '';
            document.getElementById('resultDisplay').textContent = 'Ready';
            document.getElementById('operationInfo').textContent = 'Ready';
            document.getElementById('statusInfo').textContent = '';
            document.getElementById('statusInfo').className = '';
            history = [];
            currentResult = null;
        }

        function promptPower() {
            const power = prompt('Enter power (integer):', '2');
            if (power !== null) {
                const p = parseInt(power);
                if (!isNaN(p)) {
                    operate('power');
                } else {
                    alert('Please enter a valid integer');
                }
            }
        }

        function promptSolve() {
            const b = prompt('Enter vector b (space/comma separated):');
            if (b !== null && b.trim()) {
                operate('solve');
            }
        }

        function promptLeastSquares() {
            const b = prompt('Enter vector b (space/comma separated):');
            if (b !== null && b.trim()) {
                operate('least_squares');
            }
        }

        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'c') {
                e.preventDefault();
                copyResult();
            }
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                undoLast();
            }
        });

        window.onload = function() {
            fillMatrix('A', 'zeros');
            fillMatrix('B', 'zeros');
        };
    </script>
</body>
</html>
'''

def parse_matrix(matrix_data):
    """Parse matrix from JSON or string format"""
    if isinstance(matrix_data, list):
        return np.array(matrix_data)
    if isinstance(matrix_data, str):
        try:
            lines = matrix_data.strip().split('\n')
            matrix = []
            for line in lines:
                values = re.split(r'[,\s]+', line.strip())
                values = [v for v in values if v]
                if values:
                    matrix.append([float(v) for v in values])
            if not matrix:
                return None
            row_lengths = [len(row) for row in matrix]
            if len(set(row_lengths)) > 1:
                return None
            return np.array(matrix)
        except:
            return None
    return None

def matrix_to_string(matrix):
    """Convert numpy array to formatted string"""
    if matrix is None:
        return ""
    if np.isscalar(matrix):
        return f"{matrix:.8f}"
    return np.array2string(matrix, precision=6, suppress_small=True, threshold=np.inf, max_line_width=120)

def array_to_string(arr):
    """Convert array to string for display"""
    if arr is None:
        return ""
    if np.isscalar(arr):
        return f"{arr:.6f}"
    return np.array2string(arr, precision=6, suppress_small=True, threshold=np.inf, max_line_width=120)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/fill', methods=['POST'])
def fill_matrix():
    """Fill matrix with predefined values"""
    data = request.json
    fill_type = data.get('type', 'zeros')
    rows = data.get('rows', 3)
    cols = data.get('cols', 3)
    
    try:
        if fill_type == 'identity':
            matrix = np.eye(rows, cols)
        elif fill_type == 'zeros':
            matrix = np.zeros((rows, cols))
        elif fill_type == 'ones':
            matrix = np.ones((rows, cols))
        elif fill_type == 'random':
            matrix = np.random.randn(rows, cols)
        else:
            return jsonify({'success': False, 'error': 'Unknown fill type'})
        
        return jsonify({
            'success': True,
            'matrix': matrix.tolist(),
            'str': matrix_to_string(matrix)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/operate', methods=['POST'])
def operate_matrices():
    """Perform matrix operations"""
    data = request.json
    operation = data.get('operation', '')
    matrix_a_data = data.get('matrix_a', '')
    matrix_b_data = data.get('matrix_b', None)
    
    A = parse_matrix(matrix_a_data)
    if A is None:
        return jsonify({'success': False, 'error': 'Invalid Matrix A'})
    
    B = parse_matrix(matrix_b_data) if matrix_b_data else None
    
    try:
        result = None
        info = ""
        
        if operation == 'add':
            if B is None:
                return jsonify({'success': False, 'error': 'Matrix B required'})
            if A.shape != B.shape:
                return jsonify({'success': False, 'error': f'Matrices must have same shape: {A.shape} vs {B.shape}'})
            result = A + B
            info = f"{A.shape} + {B.shape}"
        elif operation == 'subtract':
            if B is None:
                return jsonify({'success': False, 'error': 'Matrix B required'})
            if A.shape != B.shape:
                return jsonify({'success': False, 'error': f'Matrices must have same shape: {A.shape} vs {B.shape}'})
            result = A - B
            info = f"{A.shape} - {B.shape}"
        elif operation == 'multiply':
            if B is None:
                return jsonify({'success': False, 'error': 'Matrix B required'})
            if A.shape[1] != B.shape[0]:
                return jsonify({'success': False, 'error': f'Incompatible shapes: {A.shape} and {B.shape}'})
            result = A @ B
            info = f"{A.shape} × {B.shape}"
        elif operation == 'transpose':
            result = A.T
            info = f"{A.shape} → {result.shape}"
        elif operation == 'inverse':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for inverse, got {A.shape}'})
            result = np.linalg.inv(A)
            info = f"{A.shape}"
        elif operation == 'determinant':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for determinant, got {A.shape}'})
            result = np.linalg.det(A)
            info = f"det = {result:.6f}"
        elif operation == 'rank':
            result = np.linalg.matrix_rank(A)
            info = f"rank = {result}"
        elif operation == 'trace':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for trace, got {A.shape}'})
            result = np.trace(A)
            info = f"tr = {result:.6f}"
        elif operation == 'eigenvalues':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for eigenvalues, got {A.shape}'})
            eigvals = np.linalg.eigvals(A)
            result_str = "Eigenvalues:\n\n"
            for i, val in enumerate(eigvals):
                result_str += f"λ{i+1} = {val:.6f}\n"
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'eigenvectors':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for eigenvectors, got {A.shape}'})
            eigvals, eigvecs = np.linalg.eig(A)
            result_str = "Eigenvectors (columns):\n\n"
            result_str += matrix_to_string(eigvecs)
            result_str += "\n\nEigenvalues:\n"
            for i, val in enumerate(eigvals):
                result_str += f"λ{i+1} = {val:.6f}\n"
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'frobenius_norm':
            result = np.linalg.norm(A, 'fro')
            info = f"||A||_F = {result:.6f}"
        elif operation == 'infinity_norm':
            result = np.linalg.norm(A, np.inf)
            info = f"||A||_∞ = {result:.6f}"
        elif operation == 'l1_norm':
            result = np.linalg.norm(A, 1)
            info = f"||A||_1 = {result:.6f}"
        elif operation == 'condition':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for condition number, got {A.shape}'})
            result = np.linalg.cond(A)
            info = f"cond(A) = {result:.6f}"
        elif operation == 'lu':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for LU decomposition, got {A.shape}'})
            P, L, U = lu(A)
            result_str = "LU Decomposition:\n\nP (Permutation):\n"
            result_str += matrix_to_string(P)
            result_str += "\n\nL (Lower):\n"
            result_str += matrix_to_string(L)
            result_str += "\n\nU (Upper):\n"
            result_str += matrix_to_string(U)
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'qr':
            Q, R = qr(A)
            result_str = "QR Decomposition:\n\nQ (Orthogonal):\n"
            result_str += matrix_to_string(Q)
            result_str += "\n\nR (Upper):\n"
            result_str += matrix_to_string(R)
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'svd':
            U, S, Vh = svd(A, full_matrices=False)
            result_str = "SVD Decomposition:\n\nU (Left):\n"
            result_str += matrix_to_string(U)
            result_str += f"\n\nS (Singular Values):\n{array_to_string(S)}\n"
            result_str += "\nVh (Right):\n"
            result_str += matrix_to_string(Vh)
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'cholesky':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for Cholesky, got {A.shape}'})
            L = np.linalg.cholesky(A)
            result_str = "Cholesky Decomposition:\n\nL (Lower):\n"
            result_str += matrix_to_string(L)
            result_str += "\n\nL^T (Upper):\n"
            result_str += matrix_to_string(L.T)
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'matrix_exp':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for exponential, got {A.shape}'})
            result = expm(A)
            info = f"exp({A.shape})"
        elif operation == 'matrix_log':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for logarithm, got {A.shape}'})
            result = logm(A)
            info = f"log({A.shape})"
        elif operation == 'schur':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for Schur, got {A.shape}'})
            T, Z = schur(A)
            result_str = "Schur Decomposition:\n\nT (Upper Triangular):\n"
            result_str += matrix_to_string(T)
            result_str += "\n\nZ (Unitary):\n"
            result_str += matrix_to_string(Z)
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'hessenberg':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for Hessenberg, got {A.shape}'})
            H, Q = hessenberg(A, calc_q=True)
            result_str = "Hessenberg Form:\n\nH (Hessenberg):\n"
            result_str += matrix_to_string(H)
            result_str += "\n\nQ (Unitary):\n"
            result_str += matrix_to_string(Q)
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'power':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for power, got {A.shape}'})
            power = data.get('power', 2)
            if not isinstance(power, (int, float)):
                return jsonify({'success': False, 'error': 'Power must be a number'})
            result = np.linalg.matrix_power(A, int(power))
            info = f"{A.shape}^{power}"
        elif operation == 'solve':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': False, 'error': f'Matrix must be square for solving, got {A.shape}'})
            b_data = data.get('b', '')
            if not b_data:
                return jsonify({'success': False, 'error': 'Vector b required'})
            b = parse_matrix(b_data)
            if b is None:
                return jsonify({'success': False, 'error': 'Invalid vector b'})
            if b.ndim == 2 and b.shape[1] > 1:
                b = b.flatten()
            elif b.ndim == 2:
                b = b.flatten()
            if b.shape[0] != A.shape[0]:
                return jsonify({'success': False, 'error': f'b must have {A.shape[0]} elements, got {b.shape[0]}'})
            x = solve(A, b)
            return jsonify({'success': True, 'result': matrix_to_string(x), 'info': f"solution for {A.shape}"})
        elif operation == 'least_squares':
            b_data = data.get('b', '')
            if not b_data:
                return jsonify({'success': False, 'error': 'Vector b required'})
            b = parse_matrix(b_data)
            if b is None:
                return jsonify({'success': False, 'error': 'Invalid vector b'})
            if b.ndim == 2 and b.shape[1] > 1:
                b = b.flatten()
            elif b.ndim == 2:
                b = b.flatten()
            if b.shape[0] != A.shape[0]:
                return jsonify({'success': False, 'error': f'b must have {A.shape[0]} elements, got {b.shape[0]}'})
            x, resid, rank, s = lstsq(A, b)
            result_str = f"Least Squares Solution:\n{matrix_to_string(x)}\n\nResidual: {resid}\nRank: {rank}\nSingular Values: {array_to_string(s)}"
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'pseudo_inverse':
            result = pinv(A)
            info = f"{A.shape} → {result.shape}"
        elif operation == 'stats':
            flat = A.flatten()
            stats = describe(flat)
            result_str = f"Matrix Statistics ({A.size} elements):\n\n"
            result_str += f"Mean: {stats.mean:.6f}\n"
            result_str += f"Variance: {stats.variance:.6f}\n"
            result_str += f"Skewness: {stats.skewness:.6f}\n"
            result_str += f"Kurtosis: {stats.kurtosis:.6f}\n"
            result_str += f"Min: {stats.minmax[0]:.6f}\n"
            result_str += f"Max: {stats.minmax[1]:.6f}\n"
            result_str += f"Std Dev: {np.std(flat):.6f}\n"
            result_str += f"Sum: {np.sum(flat):.6f}"
            return jsonify({'success': True, 'result': result_str, 'info': f"{A.shape}"})
        elif operation == 'gram':
            G = A.T @ A
            result = G
            info = f"{A.shape} → {G.shape}"
        elif operation == 'covariance':
            C = np.cov(A, rowvar=False)
            result = C
            info = f"{A.shape} → {C.shape}"
        elif operation == 'correlation':
            C = np.corrcoef(A, rowvar=False)
            result = C
            info = f"{A.shape} → {C.shape}"
        elif operation == 'distance':
            D = squareform(pdist(A))
            result = D
            info = f"{A.shape} → {D.shape}"
        elif operation == 'is_square':
            is_sq = A.shape[0] == A.shape[1]
            return jsonify({'success': True, 'result': f"Is Square: {is_sq}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_symmetric':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Symmetric: False\nShape: {A.shape} (not square)", 'info': ""})
            is_sym = np.allclose(A, A.T)
            return jsonify({'success': True, 'result': f"Is Symmetric: {is_sym}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_skew_symmetric':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Skew-Symmetric: False\nShape: {A.shape} (not square)", 'info': ""})
            is_skew = np.allclose(A, -A.T)
            return jsonify({'success': True, 'result': f"Is Skew-Symmetric: {is_skew}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_diagonal':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Diagonal: False\nShape: {A.shape} (not square)", 'info': ""})
            is_diag = np.allclose(A - np.diag(np.diag(A)), np.zeros_like(A))
            return jsonify({'success': True, 'result': f"Is Diagonal: {is_diag}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_identity':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Identity: False\nShape: {A.shape} (not square)", 'info': ""})
            is_id = np.allclose(A, np.eye(A.shape[0]))
            return jsonify({'success': True, 'result': f"Is Identity: {is_id}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_orthogonal':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Orthogonal: False\nShape: {A.shape} (not square)", 'info': ""})
            is_orth = np.allclose(A @ A.T, np.eye(A.shape[0]))
            return jsonify({'success': True, 'result': f"Is Orthogonal: {is_orth}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_invertible':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Invertible: False\nShape: {A.shape} (not square)", 'info': ""})
            det = np.linalg.det(A)
            is_inv = not np.isclose(det, 0)
            return jsonify({'success': True, 'result': f"Is Invertible: {is_inv}\nDet: {det:.6f}", 'info': ""})
        elif operation == 'is_positive_definite':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Positive Definite: False\nShape: {A.shape} (not square)", 'info': ""})
            try:
                is_pd = np.all(np.linalg.eigvals(A) > 0)
            except:
                is_pd = False
            return jsonify({'success': True, 'result': f"Is Positive Definite: {is_pd}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_nilpotent':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Nilpotent: False\nShape: {A.shape} (not square)", 'info': ""})
            is_nil = False
            k = 0
            for i in range(1, min(A.shape[0], 5) + 1):
                if np.allclose(np.linalg.matrix_power(A, i), np.zeros_like(A)):
                    is_nil = True
                    k = i
                    break
            return jsonify({'success': True, 'result': f"Is Nilpotent: {is_nil}\nIndex: {k if is_nil else 'N/A'}", 'info': ""})
        elif operation == 'is_idempotent':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Idempotent: False\nShape: {A.shape} (not square)", 'info': ""})
            is_idem = np.allclose(A @ A, A)
            return jsonify({'success': True, 'result': f"Is Idempotent: {is_idem}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_hermitian':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Hermitian: False\nShape: {A.shape} (not square)", 'info': ""})
            is_herm = np.allclose(A, A.conj().T)
            return jsonify({'success': True, 'result': f"Is Hermitian: {is_herm}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_unitary':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Unitary: False\nShape: {A.shape} (not square)", 'info': ""})
            is_unit = np.allclose(A @ A.conj().T, np.eye(A.shape[0]))
            return jsonify({'success': True, 'result': f"Is Unitary: {is_unit}\nShape: {A.shape}", 'info': ""})
        elif operation == 'is_normal':
            if A.shape[0] != A.shape[1]:
                return jsonify({'success': True, 'result': f"Is Normal: False\nShape: {A.shape} (not square)", 'info': ""})
            is_norm = np.allclose(A @ A.T, A.T @ A)
            return jsonify({'success': True, 'result': f"Is Normal: {is_norm}\nShape: {A.shape}", 'info': ""})
        elif operation == 'size':
            return jsonify({'success': True, 'result': f"Matrix Size:\n\nRows: {A.shape[0]}\nCols: {A.shape[1]}\nElements: {A.size}", 'info': f"{A.shape}"})
        elif operation == 'elementwise_mul':
            if B is None:
                return jsonify({'success': False, 'error': 'Matrix B required'})
            if A.shape != B.shape:
                return jsonify({'success': False, 'error': f'Matrices must have same shape: {A.shape} vs {B.shape}'})
            result = A * B
            info = f"{A.shape} ⊙ {B.shape}"
        elif operation == 'elementwise_div':
            if B is None:
                return jsonify({'success': False, 'error': 'Matrix B required'})
            if A.shape != B.shape:
                return jsonify({'success': False, 'error': f'Matrices must have same shape: {A.shape} vs {B.shape}'})
            result = A / B
            info = f"{A.shape} ⊘ {B.shape}"
        elif operation == 'elementwise_power':
            power = data.get('power', 2)
            result = A ** power
            info = f"{A.shape} .^ {power}"
        elif operation == 'elementwise_abs':
            result = np.abs(A)
            info = f"|{A.shape}|"
        elif operation == 'elementwise_sqrt':
            result = np.sqrt(np.abs(A))
            info = f"√{A.shape}"
        elif operation == 'elementwise_log':
            result = np.log(np.abs(A) + 1e-10)
            info = f"log({A.shape})"
        elif operation == 'elementwise_exp':
            result = np.exp(A)
            info = f"exp({A.shape})"
        elif operation == 'elementwise_sin':
            result = np.sin(A)
            info = f"sin({A.shape})"
        elif operation == 'elementwise_cos':
            result = np.cos(A)
            info = f"cos({A.shape})"
        elif operation == 'elementwise_tan':
            result = np.tan(A)
            info = f"tan({A.shape})"
        elif operation == 'elementwise_sigmoid':
            result = 1 / (1 + np.exp(-A))
            info = f"sigmoid({A.shape})"
        else:
            return jsonify({'success': False, 'error': f'Unknown operation: {operation}'})
        
        if isinstance(result, (int, float, np.number)):
            return jsonify({'success': True, 'result': f"{result:.8f}", 'info': info})
        return jsonify({'success': True, 'result': matrix_to_string(result), 'info': info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)