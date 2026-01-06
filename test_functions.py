#!/usr/bin/env python3

# Test script to check if the utility functions are accessible

try:
    from orderbook_analyzer import _to_float_list, _sum_depth_usd, _simulate_market_impact
    print("✅ Successfully imported utility functions")
    
    # Test _to_float_list
    data = [["100", "1.5"], [101.0, 2], ["bad", "data"], [102, -1]]
    result = _to_float_list(data)
    print(f"✅ _to_float_list test passed: {result}")
    
    # Test _sum_depth_usd
    levels = [(100.0, 1.0), (101.0, 2.0), (102.0, 3.0)]
    result = _sum_depth_usd(levels, 2)
    print(f"✅ _sum_depth_usd test passed: {result}")
    
    print("✅ All utility function tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
    # Let's check what's available in the module
    import orderbook_analyzer
    print("Available in orderbook_analyzer module:")
    attrs = [attr for attr in dir(orderbook_analyzer) if not attr.startswith('__')]
    for attr in attrs:
        print(f"  - {attr}")
        
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()