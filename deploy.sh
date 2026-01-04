#!/bin/bash

# ============================================================================
# Options Analytics Engine - Quick Deploy Script
# ============================================================================

echo "================================================"
echo "Options Analytics Engine - Deployment Setup"
echo "================================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${BLUE}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo -e "${RED}Error: Python 3.9+ required${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${BLUE}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "\n${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install dependencies
echo -e "\n${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create outputs directory
echo -e "\n${BLUE}Creating outputs directory...${NC}"
mkdir -p outputs
echo -e "${GREEN}✓ Outputs directory created${NC}"

# Test Python implementation
echo -e "\n${BLUE}Testing Python implementation...${NC}"
if python advanced_options_engine.py > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Python engine working${NC}"
else
    echo -e "${RED}✗ Python engine test failed${NC}"
fi

# Compile C++ implementation
echo -e "\n${BLUE}Compiling C++ implementation...${NC}"
if command -v g++ &> /dev/null; then
    if g++ -std=c++17 -O3 -fopenmp -march=native advanced_options.cpp -o options_engine 2>/dev/null; then
        echo -e "${GREEN}✓ C++ compiled successfully${NC}"
        echo "  Run with: ./options_engine"
    else
        echo -e "${RED}✗ C++ compilation failed (OpenMP may not be available)${NC}"
        echo "  Trying without OpenMP..."
        if g++ -std=c++17 -O3 advanced_options.cpp -o options_engine 2>/dev/null; then
            echo -e "${GREEN}✓ C++ compiled (without OpenMP)${NC}"
        fi
    fi
else
    echo -e "${RED}✗ g++ not found. Install with: brew install gcc (macOS) or apt install g++ (Linux)${NC}"
fi

# Deployment options
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"

echo -e "\n${BLUE}Next Steps:${NC}"
echo ""
echo "1. Test the dashboard locally:"
echo "   ${GREEN}streamlit run options_dashboard.py${NC}"
echo ""
echo "2. Deploy to Streamlit Cloud (FREE):"
echo "   a. Push to GitHub:"
echo "      ${GREEN}git init${NC}"
echo "      ${GREEN}git add .${NC}"
echo "      ${GREEN}git commit -m 'Advanced Options Analytics Engine'${NC}"
echo "      ${GREEN}git remote add origin YOUR_GITHUB_URL${NC}"
echo "      ${GREEN}git push -u origin main${NC}"
echo ""
echo "   b. Visit: https://share.streamlit.io"
echo "   c. Connect your GitHub repo"
echo "   d. Select: options_dashboard.py"
echo "   e. Click Deploy!"
echo ""
echo "3. Run C++ engine:"
echo "   ${GREEN}./options_engine${NC}"
echo ""
echo "4. Run Python engine:"
echo "   ${GREEN}python advanced_options_engine.py${NC}"
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${BLUE}Project Structure:${NC}"
echo "  advanced_options_engine.py  - Core Python library"
echo "  options_dashboard.py        - Streamlit web app"
echo "  advanced_options.cpp        - C++ implementation"
echo "  requirements.txt            - Python dependencies"
echo "  README.md                   - Documentation"
echo "  PROJECT_ASSESSMENT.md       - Detailed analysis"
echo ""
echo -e "${GREEN}For deployment help, see PROJECT_ASSESSMENT.md${NC}"
echo ""
