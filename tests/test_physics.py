import numpy as np
import pytest
from CBm0.physics import cp_molar_shomate_field, R_u

def test_constants():
    """Check universal gas constant."""
    assert R_u == 8.314

def test_shomate_cp_n2():
    """Test specific heat calculation for N2 at standard conditions."""
    # Create a dummy temperature field
    T = np.array([[300.0, 1000.0], [2000.0, 3000.0]])
    
    # Calculate Cp for N2
    cp = cp_molar_shomate_field('N2', T)
    
    # Expected range for N2 Cp [J/mol/K]
    # At 300K ~ 29.12
    # At 1000K ~ 32.7
    
    assert cp.shape == T.shape
    assert np.all(cp > 28.0)
    assert np.all(cp < 40.0)
    
    # Check specific value at 300K (approx)
    assert np.isclose(cp[0, 0], 29.12, atol=0.5)

def test_shomate_cp_invalid_species():
    """Test fallback or behavior for unknown species (defaults to H2O in current impl)."""
    T = np.array([[300.0]])
    # The current implementation defaults to H2O for unknown keys in _shomate_pair logic 
    # if not explicitly handled, but let's just test a known one to be safe or check the code.
    # Looking at code: else -> H2O.
    cp = cp_molar_shomate_field('UNKNOWN', T)
    cp_h2o = cp_molar_shomate_field('H2O', T)
    assert np.allclose(cp, cp_h2o)
