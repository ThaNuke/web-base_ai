"""
Generate table report from model importance results JSON
Shows: ลำดับ, Score All, scScore Without Freq, Impact
"""

import json
from pathlib import Path

def generate_table_report():
    """Generate table report in HTML format"""
    
    # Load JSON results
    json_file = Path("model_importance_results_senior_project.json")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_results = data["test_results"]
    
    # Create HTML report
    html_content = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Importance Report - Frequency Impact Analysis</title>
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
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            font-size: 0.9em;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        .summary-item {
            text-align: center;
        }
        .summary-item h3 {
            color: #667eea;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        .summary-item .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            font-size: 0.95em;
        }
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            position: sticky;
            top: 0;
        }
        th {
            padding: 15px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #ddd;
        }
        td {
            padding: 12px 15px;
            text-align: center;
            border: 1px solid #ddd;
        }
        tbody tr {
            transition: background-color 0.2s;
        }
        tbody tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tbody tr:hover {
            background-color: #f0f0f0;
        }
        .row-num {
            font-weight: 500;
            color: #667eea;
            width: 60px;
        }
        .score {
            background-color: #e8f4f8;
            font-weight: 500;
        }
        .score-without {
            background-color: #fff3e0;
            font-weight: 500;
        }
        .impact {
            background-color: #e8f5e9;
            font-weight: 600;
            color: #2e7d32;
        }
        .controls {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: all 0.3s;
        }
        .btn-download {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-download:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .notes {
            background: #f5f5f5;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin-top: 20px;
            border-radius: 4px;
            font-size: 0.9em;
            color: #555;
        }
        .notes h4 {
            color: #667eea;
            margin-bottom: 8px;
        }
        @media print {
            body {
                background: white;
            }
            .container {
                box-shadow: none;
            }
            .controls {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Model Importance - Frequency Impact Analysis</h1>
        <p class="subtitle">Ablation Study Results - Dataset Senior Project</p>
        
        <div class="summary">
            <div class="summary-item">
                <h3>ทั้งหมด</h3>
                <div class="value">""" + str(len(test_results)) + """</div>
            </div>
            <div class="summary-item">
                <h3>Real Images</h3>
                <div class="value">""" + str(data["real_images"]) + """</div>
            </div>
            <div class="summary-item">
                <h3>AI Images</h3>
                <div class="value">""" + str(data["ai_images"]) + """</div>
            </div>
            <div class="summary-item">
                <h3>Avg Impact (Freq)</h3>
                <div class="value">""" + f"{data['summary']['average_impacts']['frequency']:.4f}" + """</div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-download" onclick="downloadCSV()">📥 Download CSV</button>
            <button class="btn btn-download" onclick="window.print()">🖨️ Print</button>
        </div>

        <table id="resultTable">
            <thead>
                <tr>
                    <th>ลำดับ</th>
                    <th>ชื่อไฟล์</th>
                    <th>ประเภท</th>
                    <th>Score All<br/>(Baseline)</th>
                    <th>scScore Without Freq<br/>(After Removal)</th>
                    <th>Impact<br/>(Frequency)</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add table rows
    for idx, result in enumerate(test_results, 1):
        filename = result["filename"]
        is_real = "🟩 Real" if result["is_real"] else "🔴 AI"
        baseline = result["baseline"]
        frequency_impact = result["impacts"]["frequency"]
        # Prefer the real "without" score (direction-aware). Fallback to legacy baseline-impact.
        if isinstance(result.get("without"), dict) and "frequency" in result["without"]:
            score_without_freq = result["without"]["frequency"]
        else:
            score_without_freq = baseline - frequency_impact
        
        html_content += f"""
                <tr>
                    <td class="row-num">{idx}</td>
                    <td style="text-align: left; font-family: monospace; font-size: 0.85em;">{filename}</td>
                    <td>{is_real}</td>
                    <td class="score">{baseline:.2f}</td>
                    <td class="score-without">{score_without_freq:.2f}</td>
                    <td class="impact">{frequency_impact:.2f}</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>

        <div class="notes">
            <h4>📊 สูตรการคำนวณ:</h4>
            <ul style="margin-left: 20px;">
                <li><strong>Score All</strong> = Baseline (ใช้ทั้ง 4 model)</li>
                <li><strong>Score Without Freq</strong> = คะแนน ensemble เมื่อปิดโมเดล Frequency (คำนวณใหม่จากโมเดลที่เหลือ)</li>
                <li><strong>Impact</strong> = |Score All - Score Without Freq|</li>
            </ul>
        </div>

        <div class="notes">
            <h4>💡 การอ่านข้อมูล:</h4>
            <ul style="margin-left: 20px;">
                <li>ยิ่ง Impact สูง แสดงว่า Frequency ส่งผลต่อคะแนน ensemble มาก</li>
                <li>Impact ในไฟล์ผลลัพธ์เป็นค่าสัมบูรณ์ จึงไม่บอกทิศทางว่าปิดแล้วคะแนนเพิ่มหรือลด (ดูได้จาก Score Without)</li>
            </ul>
        </div>
    </div>

    <script>
        function downloadCSV() {
            const table = document.getElementById('resultTable');
            let csv = [];
            
            // Headers
            const headers = [];
            table.querySelectorAll('thead th').forEach(th => {
                headers.push('"' + th.textContent.replace(/\\n/g, ' ') + '"');
            });
            csv.push(headers.join(','));
            
            // Rows
            table.querySelectorAll('tbody tr').forEach(tr => {
                const row = [];
                tr.querySelectorAll('td').forEach(td => {
                    row.push('"' + td.textContent.trim() + '"');
                });
                csv.push(row.join(','));
            });
            
            // Download
            const csvContent = csv.join('\\n');
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'model_importance_frequency_report.csv';
            link.click();
        }
    </script>
</body>
</html>
"""
    
    # Save HTML report
    output_file = Path("model_importance_report.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML Report generated: {output_file}")
    print(f"Total rows: {len(test_results)}")
    print()
    
    # Also create a simple CSV version
    create_csv_report(test_results, data)


def create_csv_report(test_results, data):
    """Create CSV report"""
    import csv
    
    csv_file = Path("model_importance_report.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'ลำดับ', 'ชื่อไฟล์', 'ประเภท', 
            'Score All', 'scScore Without Freq', 'Impact (Frequency)'
        ])
        
        # Data rows
        for idx, result in enumerate(test_results, 1):
            filename = result["filename"]
            is_real = "Real" if result["is_real"] else "AI"
            baseline = result["baseline"]
            frequency_impact = result["impacts"]["frequency"]
            if isinstance(result.get("without"), dict) and "frequency" in result["without"]:
                score_without_freq = result["without"]["frequency"]
            else:
                score_without_freq = baseline - frequency_impact
            
            writer.writerow([
                idx, filename, is_real,
                f"{baseline:.2f}", f"{score_without_freq:.2f}", f"{frequency_impact:.2f}"
            ])
    
    print(f"CSV Report generated: {csv_file}")


if __name__ == "__main__":
    generate_table_report()
