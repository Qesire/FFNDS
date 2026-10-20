"""Run the original HiF4 Smooth+RTN W4A4 baseline with dense BF16 GEMMs."""
from run_distill import main

if __name__ == "__main__":
    main("D-Q")
