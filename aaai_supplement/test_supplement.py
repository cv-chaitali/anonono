#!/usr/bin/env python3
"""
Test script to verify the FineScope AAAI supplement structure and functionality.
This script checks that all components work correctly without external dependencies.
"""

import os
import sys
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    required_files = [
        "README.md",
        "requirements.txt", 
        "setup_env.sh",
        "main.py",
        "sae.py",
        "curation.py",
        "data_generator.py",
        "config.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_imports():
    """Test that the code structure is valid."""
    try:
        # Test config imports
        sys.path.insert(0, '.')
        from config import SaeConfig, TrainingConfig, CurationConfig
        print("✅ Config modules import correctly")
        
        # Test SAE class structure
        from sae import Sae, EncoderOutput, ForwardOutput, train_sae, extract_embeddings
        print("✅ SAE module structure is valid")
        
        # Test curation structure
        from curation import run_curation_pipeline, curate_dataset, evaluate_curation_quality
        print("✅ Curation module structure is valid")
        
        # Test data generator structure
        from data_generator import create_activation_dataset, create_dummy_activations
        print("✅ Data generator module structure is valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_creation():
    """Test that configurations can be created properly."""
    try:
        from config import SaeConfig, TrainingConfig, CurationConfig
        
        # Test SAE config
        sae_cfg = SaeConfig(expansion_factor=32, k=32)
        assert sae_cfg.expansion_factor == 32
        assert sae_cfg.k == 32
        print("✅ SAE config creation works")
        
        # Test training config
        train_cfg = TrainingConfig(batch_size=32, num_epochs=10)
        assert train_cfg.batch_size == 32
        assert train_cfg.num_epochs == 10
        print("✅ Training config creation works")
        
        # Test curation config
        curation_cfg = CurationConfig(top_k_selection=100, use_clustering=True)
        assert curation_cfg.top_k_selection == 100
        assert curation_cfg.use_clustering == True
        print("✅ Curation config creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation error: {e}")
        return False

def test_readme_content():
    """Test that README contains required sections."""
    try:
        with open("README.md", "r") as f:
            content = f.read()
        
        required_sections = [
            "FineScope",
            "Overview", 
            "Installation",
            "Quick Start",
            "Usage Examples",
            "Configuration",
            "Requirements",
            "Anonymization"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing README sections: {missing_sections}")
            return False
        else:
            print("✅ README contains all required sections")
            return True
            
    except Exception as e:
        print(f"❌ README test error: {e}")
        return False

def test_requirements():
    """Test that requirements.txt contains necessary packages."""
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        required_packages = [
            "torch",
            "numpy", 
            "scikit-learn",
            "safetensors",
            "tqdm",
            "matplotlib",
            "seaborn"
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in content:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements.txt: {missing_packages}")
            return False
        else:
            print("✅ Requirements.txt contains all necessary packages")
            return True
            
    except Exception as e:
        print(f"❌ Requirements test error: {e}")
        return False

def test_setup_script():
    """Test that setup script exists and is executable."""
    try:
        if not os.path.exists("setup_env.sh"):
            print("❌ setup_env.sh not found")
            return False
        
        # Check if executable
        if not os.access("setup_env.sh", os.X_OK):
            print("❌ setup_env.sh is not executable")
            return False
        
        # Check content
        with open("setup_env.sh", "r") as f:
            content = f.read()
        
        if "finescope_env" not in content or "requirements.txt" not in content:
            print("❌ setup_env.sh content is incomplete")
            return False
        
        print("✅ Setup script is valid and executable")
        return True
        
    except Exception as e:
        print(f"❌ Setup script test error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing FineScope AAAI Supplement")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Code Imports", test_imports),
        ("Config Creation", test_config_creation),
        ("README Content", test_readme_content),
        ("Requirements", test_requirements),
        ("Setup Script", test_setup_script)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! The supplement is ready for submission.")
        print("\nTo use the supplement:")
        print("1. Run: ./setup_env.sh")
        print("2. Activate: source finescope_env/bin/activate")
        print("3. Run demo: python main.py")
        return True
    else:
        print("❌ Some tests failed. Please check the structure.")
        return False

if __name__ == "__main__":
    main() 