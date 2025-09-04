#!/usr/bin/env python3
"""
Run all unit tests for TinyMPC subfunctions
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import all test modules
from test_forward_pass import TestForwardPass
from test_slack_update import TestSlackUpdate
from test_dual_update import TestDualUpdate
from test_backward_pass import TestBackwardPass
from test_linear_cost import TestLinearCost
from test_termination import TestTerminationCheck
from test_integration import TestTinyMPCIntegration


def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestForwardPass,
        TestSlackUpdate,
        TestDualUpdate,
        TestBackwardPass,
        TestLinearCost,
        TestTerminationCheck,
        TestTinyMPCIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def main():
    """Run all tests with detailed output"""
    print("TinyMPC Subfunction Unit Test Suite")
    print("=" * 50)
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  ❌ {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0] if 'AssertionError:' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for test, traceback in result.errors:
            error_msg = traceback.split('\\n')[-2] if len(traceback.split('\\n')) > 1 else 'Unknown error'
            print(f"  💥 {test}: {error_msg}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("TinyMPC subfunctions are working correctly:")
        print("  ✓ Forward Pass - Control and state prediction")
        print("  ✓ Slack Update - Constraint handling")
        print("  ✓ Dual Update - ADMM dual variable updates")
        print("  ✓ Linear Cost - Reference tracking and penalties")
        print("  ✓ Backward Pass - Riccati recursion")
        print("  ✓ Termination - Convergence checking")
        print("  ✓ Integration - Complete solver generation")
    else:
        print("\n⚠️ Some tests failed. Check details above.")
    
    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)