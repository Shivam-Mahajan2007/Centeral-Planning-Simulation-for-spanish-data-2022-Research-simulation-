#!/usr/bin/env python3
"""
test_macge.py
-------------
Simple test script for the MA-CGE implementation.

Tests calibration and basic simulation functionality.
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_calibration():
    """Test MA-CGE calibration."""
    try:
        from CGE.ma_calibration import calibrate_ma
        
        logger.info("Testing MA-CGE calibration...")
        state = calibrate_ma(seed=42)
        
        # Basic checks
        assert state.n > 0, "Number of sectors should be positive"
        assert state.n_households == 4, "Should have 4 households"
        assert state.n_firms == 5, "Should have 5 firms"
        assert len(state.P) == state.n, "Price vector length should match sectors"
        assert len(state.K) == state.n, "Capital vector length should match sectors"
        assert len(state.X) == state.n, "Output vector length should match sectors"
        
        logger.info(f"✓ Calibration successful: {state.n} sectors, {state.n_households} households")
        logger.info(f"  Initial GDP: {np.dot(state.P, state.X):.2f}")
        logger.info(f"  Average markup: {np.mean(state.markup):.4f}")
        
        return state
        
    except Exception as e:
        logger.error(f"✗ Calibration failed: {e}")
        return None

def test_simulation(state):
    """Test basic simulation functionality."""
    try:
        from CGE.ma_simulation import run_quarter_ma
        
        logger.info("Testing single quarter simulation...")
        
        # Store initial values
        initial_gdp = np.dot(state.P, state.X)
        initial_capital = state.K.copy()
        
        # Run one quarter
        state_new = run_quarter_ma(state)
        
        # Basic checks
        assert state_new.t == 1, "Time should advance by 1"
        assert len(state_new.history) == 1, "Should have one history entry"
        assert state_new.K.sum() > 0, "Capital should remain positive"
        
        # Check GDP change
        new_gdp = np.dot(state_new.P, state_new.X)
        logger.info(f"✓ Simulation successful")
        logger.info(f"  Initial GDP: {initial_gdp:.2f}")
        logger.info(f"  New GDP: {new_gdp:.2f}")
        logger.info(f"  GDP change: {((new_gdp/initial_gdp - 1) * 100):.2f}%")
        
        return state_new
        
    except Exception as e:
        logger.error(f"✗ Simulation failed: {e}")
        return None

def test_comparison():
    """Test comparison functionality."""
    try:
        from CGE.compare import run_comparison
        
        logger.info("Testing comparison functionality...")
        
        # Run short comparison
        df = run_comparison(n_quarters=2, seed=42)
        
        # Basic checks
        assert len(df) > 0, "Should generate comparison data"
        assert 'metric' in df.columns, "Should have metric column"
        assert 'CIKP' in df.columns, "Should have CIKP column"
        assert 'MACGE' in df.columns, "Should have MACGE column"
        
        logger.info(f"✓ Comparison successful: {len(df)} observations")
        
        return df
        
    except Exception as e:
        logger.error(f"✗ Comparison failed: {e}")
        return None

def main():
    """Run all tests."""
    logger.info("Starting MA-CGE system tests...")
    
    # Test calibration
    state = test_calibration()
    if state is None:
        return False
    
    # Test simulation
    state = test_simulation(state)
    if state is None:
        return False
    
    # Test comparison (optional - may take longer)
    try:
        df = test_comparison()
        if df is not None:
            logger.info("All tests passed! ✓")
            return True
    except Exception as e:
        logger.warning(f"Comparison test failed (may be expected): {e}")
        logger.info("Core tests passed! ✓")
        return True
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
