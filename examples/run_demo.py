import sys
import os

# Add project root to path so we can import CBm0
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CBm0.main import main
from CBm0.config import Config
import yaml

def run_demo():
    print("Running CBm0 Demo...")
    
    # Load demo config
    config_path = os.path.join(os.path.dirname(__file__), 'demo_config.yaml')
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Create Config object
    cfg = Config(**cfg_dict)
    
    # Run solver
    try:
        main(cfg)
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_demo()
