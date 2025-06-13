# Skore Hub Demo Project

This repository provides a hands-on introduction to [Skore Hub](https://probabl.ai/skore). It contains a sample notebook that demonstrates how to create a project, generate experiment reports, and upload them to Skore Hub.

## Prerequisites

Ensure you have Python 3.10+ installed on your system.

## Setup

1. Clone the repository and navigate to the project directory:
```bash
git clone git@github.com:probabl-ai/skore-demo-project.git
cd skore-demo-project
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies and authenticate with Skore Hub:
```bash
pip install -r requirements.txt
skore-hub-login
```

## Usage

You have two options to run the demo:

### Option 1: Jupyter Notebook
Execute the cells in `notebooks/demo.ipynb`

### Option 2: Command Line
```bash
python demo.py --tenant="<your-tenant>" --name="demo"
```

Replace `<your-tenant>` with your actual tenant name.

## Additional Resources

- [Skore Library](https://github.com/probabl-ai/skore) - Learn more about the core Skore functionality
- [Probabl Website](https://probabl.ai/) - Discover our complete product suite
