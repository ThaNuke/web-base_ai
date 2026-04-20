# PowerShell Script for System Testing
# รัน system-level test ทั้งหมดสำหรับ Windows

param(
    [ValidateSet("single", "batch")]
    [string]$TestSuite = "single",
    [int]$NumRuns = 3,
    [string]$ServerUrl = "http://localhost:8001"
)

$ErrorActionPreference = "Stop"

# Colors function
function Write-ColorOutput($color, $message) {
    Write-Host $message -ForegroundColor $color
}

# Banner
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  TruPic API - System Level Testing (Windows)                ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-ColorOutput Green "✓ Python found: $pythonVersion"
} catch {
    Write-ColorOutput Red "✗ Python is not installed or not in PATH"
    exit 1
}

# Check dependencies
Write-Host ""
Write-ColorOutput Blue "ℹ Checking dependencies..."

$checkDeps = @"
import sys
try:
    import requests
    import PIL
    print("OK")
except ImportError as e:
    print(f"MISSING: {e}")
    sys.exit(1)
"@

$result = python -c $checkDeps 2>&1
if ($result -eq "OK") {
    Write-ColorOutput Green "✓ All dependencies available"
} else {
    Write-ColorOutput Yellow "Installing required packages..."
    pip install requests pillow
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Failed to install dependencies"
        exit 1
    }
}

# Show configuration
Write-Host ""
Write-ColorOutput Blue "Configuration:"
Write-Host "  Test Type: $TestSuite"
if ($TestSuite -eq "batch") {
    Write-Host "  Number of Runs: $NumRuns"
}
Write-Host "  Server URL: $ServerUrl"
Write-Host ""

# Check if server is running
Write-ColorOutput Blue "Checking if server is running..."

try {
    $response = (New-Object Net.WebClient).DownloadString("$ServerUrl/api/health") 2>&1
    Write-ColorOutput Green "✓ Server is running"
} catch {
    Write-ColorOutput Red "✗ Cannot connect to server at $ServerUrl"
    Write-Host ""
    Write-Host "Please start the backend server first:"
    Write-Host "  cd backend"
    Write-Host "  python main.py"
    Write-Host ""
    exit 1
}

Write-Host ""

# Run tests
if ($TestSuite -eq "batch") {
    Write-ColorOutput Blue "Running $NumRuns test iterations..."
    python batch_test_runner.py -n $NumRuns -u $ServerUrl
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Batch test runner failed"
        exit 1
    }
} else {
    Write-ColorOutput Blue "Running single system-level test..."
    python test_system_level.py $ServerUrl
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "Test failed"
        exit 1
    }
}

# Analyze results
Write-Host ""
Write-ColorOutput Blue "Analyzing performance..."
python analyze_performance.py

# Summary
Write-Host ""
Write-ColorOutput Green "✅ Testing complete!"
Write-Host ""
Write-ColorOutput Yellow "📊 Generated files:"
Write-Host "  - test_system_level_report.json"
if ($TestSuite -eq "batch") {
    Write-Host "  - batch_test_report.json"
}
Write-Host "  - performance_report.csv"
Write-Host ""
Write-ColorOutput Yellow "📖 For detailed information, see SYSTEM_TEST_GUIDE.md"
Write-Host ""

# Optional: Open report in default json viewer
$reportExists = Test-Path "test_system_level_report.json"
if ($reportExists) {
    $response = Read-Host "Open report file? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Invoke-Item "test_system_level_report.json"
    }
}
