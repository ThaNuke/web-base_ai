#!/bin/bash
# Quick Start Script for System Testing
# รัน system-level test ทั้งหมดอย่างรวดเร็ว

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  TruPic API - System Level Testing Quick Start              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python is not installed${NC}"
    exit 1
fi

echo -e "${BLUE}ℹ Checking dependencies...${NC}"

# Check required packages
python <<EOF
import sys
try:
    import requests
    import PIL
    print("✓ All dependencies are available")
except ImportError as e:
    print(f"✗ Missing: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install requests pillow
fi

# Get options
TEST_SUITE="${1:-single}"  # single or batch
NUM_RUNS="${2:-3}"
SERVER_URL="${3:-http://localhost:8001}"

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Test Type: $TEST_SUITE"
if [ "$TEST_SUITE" = "batch" ]; then
    echo "  Number of Runs: $NUM_RUNS"
fi
echo "  Server URL: $SERVER_URL"
echo ""

# Check if server is running
echo -e "${BLUE}Checking if server is running...${NC}"
if python -c "import requests; requests.get('$SERVER_URL/api/health', timeout=2)" 2>/dev/null; then
    echo -e "${GREEN}✓ Server is running${NC}"
else
    echo -e "${RED}✗ Cannot connect to server at $SERVER_URL${NC}"
    echo ""
    echo "Please start the backend server first:"
    echo "  cd backend"
    echo "  python main.py"
    exit 1
fi

echo ""

# Run tests
if [ "$TEST_SUITE" = "batch" ]; then
    echo -e "${BLUE}Running ${NUM_RUNS} test iterations...${NC}"
    python batch_test_runner.py -n $NUM_RUNS -u $SERVER_URL
else
    echo -e "${BLUE}Running single system-level test...${NC}"
    python test_system_level.py $SERVER_URL
fi

# Analyze results
echo ""
echo -e "${BLUE}Analyzing performance...${NC}"
python analyze_performance.py

echo ""
echo -e "${GREEN}✅ Testing complete!${NC}"
echo ""
echo "📊 Generated files:"
echo "  - test_system_level_report.json"
if [ "$TEST_SUITE" = "batch" ]; then
    echo "  - batch_test_report.json"
fi
echo "  - performance_report.csv"
echo ""
echo "📖 For detailed information, see SYSTEM_TEST_GUIDE.md"
echo ""
