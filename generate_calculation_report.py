"""
Generate detailed calculation report with formulas
Shows all calculations for Average Impact and Raw Weight
"""

import json
from pathlib import Path

def generate_calculation_report():
    """Generate detailed calculation report"""
    
    # Load JSON results
    json_file = Path("model_importance_results_senior_project.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_results = data["test_results"]
    summary = data["summary"]
    
    # Create HTML report
    html_content = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Importance - Detailed Calculations</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2em;
        }
        .formula-section {
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            font-family: 'Courier New', monospace;
        }
        .formula-title {
            color: #667eea;
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .formula {
            background: #1a252f;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin-bottom: 10px;
            overflow-x: auto;
            font-size: 0.95em;
            line-height: 1.6;
        }
        .calculation {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .calc-box {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .calc-box h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        .calc-steps {
            background: white;
            padding: 15px;
            border-radius: 4px;
            font-size: 0.9em;
            line-height: 1.8;
            font-family: 'Courier New', monospace;
        }
        .step {
            margin-bottom: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .step-title {
            color: #667eea;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .step-content {
            color: #333;
            margin-left: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        th {
            padding: 12px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #ddd;
        }
        td {
            padding: 10px;
            text-align: center;
            border: 1px solid #ddd;
        }
        tbody tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tbody tr:hover {
            background-color: #f0f0f0;
        }
        .copy-section {
            background: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #2196F3;
        }
        .copy-title {
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
        }
        .copy-content {
            background: white;
            padding: 15px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            user-select: all;
            cursor: pointer;
            border: 1px solid #2196F3;
            position: relative;
        }
        .copy-btn {
            position: absolute;
            top: 5px;
            right: 5px;
            padding: 5px 10px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }
        .copy-btn:hover {
            background: #1976D2;
        }
        .impact-table-container {
            overflow-x: auto;
            margin-bottom: 30px;
        }
        .model-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .model-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .model-card h4 {
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        .model-card .value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .model-card .weight {
            font-size: 1.3em;
            opacity: 0.9;
        }
        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Model Importance - Detailed Calculations</h1>
        
        <!-- Formula Section -->
        <div class="formula-section">
            <div class="formula-title">📐 สูตรการคำนวณ</div>
            
            <div class="formula">
                <strong>Average Impact (เฉพาะ Frequency):</strong><br/>
                Avg Impact = (1/n) × Σ(Score_all - Score_without_freq)<br/>
                Avg Impact = (1/300) × (""" + " + ".join([f"{r['impacts']['frequency']:.2f}" for r in test_results[:5]]) + """ + ... + """ + f"{test_results[-1]['impacts']['frequency']:.2f}" + """)
            </div>
            
            <div class="formula">
                <strong>Raw Weight (Importance %):</strong><br/>
                Raw Weight_frequency = (Impact_frequency / Σ Impact) × 100%<br/>
                Raw Weight_frequency = """ + f"{summary['average_impacts']['frequency']:.4f}" + """ / """ + f"{summary['total_impact']:.4f}" + """ × 100%
            </div>
        </div>
        
        <!-- Calculation Steps -->
        <div class="calculation">
            <div class="calc-box">
                <h3>🔢 Step 1: Calculate Average Impact (Frequency)</h3>
                <div class="calc-steps">
                    <div class="step">
                        <div class="step-title">Collect all Frequency impacts:</div>
                        <div class="step-content">
                            """ + ", ".join([f"{r['impacts']['frequency']:.2f}" for r in test_results[:10]]) + """, ... (300 samples)
                        </div>
                    </div>
                    <div class="step">
                        <div class="step-title">Sum all impacts:</div>
                        <div class="step-content">
                            Σ = """ + f"{sum([r['impacts']['frequency'] for r in test_results]):.2f}" + """
                        </div>
                    </div>
                    <div class="step">
                        <div class="step-title">Divide by count:</div>
                        <div class="step-content">
                            """ + f"{sum([r['impacts']['frequency'] for r in test_results]):.2f}" + """ ÷ 300 = <br/>
                            <strong>""" + f"{summary['average_impacts']['frequency']:.4f}" + """</strong>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="calc-box">
                <h3>📈 Step 2: Calculate Raw Weight</h3>
                <div class="calc-steps">
                    <div class="step">
                        <div class="step-title">Get all average impacts:</div>
                        <div class="step-content">
                            Frequency: """ + f"{summary['average_impacts']['frequency']:.4f}" + """<br/>
                            Pixel: """ + f"{summary['average_impacts']['pixel']:.4f}" + """<br/>
                            ELA: """ + f"{summary['average_impacts']['ela']:.4f}" + """<br/>
                            Xception: """ + f"{summary['average_impacts']['xception']:.4f}" + """
                        </div>
                    </div>
                    <div class="step">
                        <div class="step-title">Sum total impact:</div>
                        <div class="step-content">
                            Total = """ + f"{summary['total_impact']:.4f}" + """
                        </div>
                    </div>
                    <div class="step">
                        <div class="step-title">Calculate weight:</div>
                        <div class="step-content">
                            """ + f"{summary['average_impacts']['frequency']:.4f}" + """ ÷ """ + f"{summary['total_impact']:.4f}" + """ × 100% = <br/>
                            <strong>""" + f"{summary['weights']['frequency']:.2f}" + """%</strong>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Model Comparison -->
        <h2 style="margin-bottom: 20px; color: #333;">🏆 Model Importance Ranking</h2>
        <div class="model-comparison">
            <div class="model-card">
                <h4>1️⃣ FREQUENCY</h4>
                <div class="value">""" + f"{summary['average_impacts']['frequency']:.4f}" + """</div>
                <div class="weight">""" + f"{summary['weights']['frequency']:.2f}" + """%</div>
            </div>
            <div class="model-card">
                <h4>2️⃣ PIXEL</h4>
                <div class="value">""" + f"{summary['average_impacts']['pixel']:.4f}" + """</div>
                <div class="weight">""" + f"{summary['weights']['pixel']:.2f}" + """%</div>
            </div>
            <div class="model-card">
                <h4>3️⃣ ELA</h4>
                <div class="value">""" + f"{summary['average_impacts']['ela']:.4f}" + """</div>
                <div class="weight">""" + f"{summary['weights']['ela']:.2f}" + """%</div>
            </div>
            <div class="model-card">
                <h4>4️⃣ XCEPTION</h4>
                <div class="value">""" + f"{summary['average_impacts']['xception']:.4f}" + """</div>
                <div class="weight">""" + f"{summary['weights']['xception']:.2f}" + """%</div>
            </div>
        </div>
        
        <!-- Summary Table -->
        <h2 style="margin-bottom: 20px; color: #333;">📋 Summary Table</h2>
        <div class="impact-table-container">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Average Impact</th>
                        <th>Weight (%)</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add rows for each model
    for model in ['frequency', 'pixel', 'ela', 'xception']:
        html_content += f"""
                    <tr>
                        <td><strong>{model.upper()}</strong></td>
                        <td>{summary['average_impacts'][model]:.4f}</td>
                        <td>{summary['weights'][model]:.2f}%</td>
                    </tr>
"""
    
    html_content += f"""
                    <tr style="background: #f0f0f0; font-weight: bold;">
                        <td>TOTAL</td>
                        <td>{summary['total_impact']:.4f}</td>
                        <td>100.00%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <!-- Copy Section -->
        <div class="copy-section">
            <div class="copy-title">📋 Copy These Values for Your Report:</div>
            <div class="copy-content" onclick="navigator.clipboard.writeText(this.innerText); alert('Copied!');">
Average Impact (Frequency) = {summary['average_impacts']['frequency']:.4f}
Raw Weight (Frequency) = {summary['weights']['frequency']:.2f}%

Average Impact (Pixel) = {summary['average_impacts']['pixel']:.4f}
Raw Weight (Pixel) = {summary['weights']['pixel']:.2f}%

Average Impact (ELA) = {summary['average_impacts']['ela']:.4f}
Raw Weight (ELA) = {summary['weights']['ela']:.2f}%

Average Impact (Xception) = {summary['average_impacts']['xception']:.4f}
Raw Weight (Xception) = {summary['weights']['xception']:.2f}%

Total Impact = {summary['total_impact']:.4f}
            </div>
            <p style="margin-top: 10px; color: #666; font-size: 0.9em;">💡 Click on the box above to copy all values</p>
        </div>
        
        <!-- Frequency Impact Details -->
        <h2 style="margin-bottom: 20px; color: #333;">📊 Frequency Impact Details (All 300 Samples)</h2>
        <div class="impact-table-container">
            <table id="detailTable">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Image</th>
                        <th>Type</th>
                        <th>Score All</th>
                        <th>Score Without Freq</th>
                        <th>Frequency Impact</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add details for each image
    for idx, result in enumerate(test_results, 1):
        img_type = "Real" if result['is_real'] else "AI"
        score_all = result['baseline']
        freq_impact = result['impacts']['frequency']
        if isinstance(result.get("without"), dict) and "frequency" in result["without"]:
            score_without = result["without"]["frequency"]
        else:
            score_without = score_all - freq_impact
        
        html_content += f"""
                    <tr>
                        <td>{idx}</td>
                        <td style="font-family: monospace; font-size: 0.85em; text-align: left;">{result['filename']}</td>
                        <td>{img_type}</td>
                        <td>{score_all:.2f}</td>
                        <td>{score_without:.2f}</td>
                        <td><strong>{freq_impact:.2f}</strong></td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <script>
            // Simple table search
            function filterTable() {
                const input = document.getElementById('searchInput');
                const filter = input.value.toLowerCase();
                const table = document.getElementById('detailTable');
                const tr = table.getElementsByTagName('tr');
                
                for (let i = 1; i < tr.length; i++) {
                    const td = tr[i].getElementsByTagName('td');
                    let found = false;
                    for (let j = 0; j < td.length; j++) {
                        if (td[j].innerText.toLowerCase().indexOf(filter) > -1) {
                            found = true;
                            break;
                        }
                    }
                    tr[i].style.display = found ? '' : 'none';
                }
            }
        </script>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    output_file = Path("model_importance_calculations.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Calculation Report generated: {output_file}")
    print()
    print("Summary Statistics:")
    print(f"   Average Impact (Frequency): {summary['average_impacts']['frequency']:.4f}")
    print(f"   Raw Weight (Frequency): {summary['weights']['frequency']:.2f}%")
    print(f"   Total Impact: {summary['total_impact']:.4f}")


if __name__ == "__main__":
    generate_calculation_report()
