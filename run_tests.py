#!/usr/bin/env python3
"""
Test Runner for Intelligent Automation Engine

This script provides a comprehensive test runner for all components
of the intelligent automation engine with various testing options.
"""

import unittest
import sys
import os
import time
import argparse
from typing import List, Optional
import coverage

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import test modules
from tests.test_parallel_execution import create_test_suite as create_parallel_suite
from tests.test_scheduling import create_test_suite as create_scheduling_suite
from tests.test_integration import create_integration_test_suite


class TestRunner:
    """Comprehensive test runner with multiple options"""
    
    def __init__(self):
        self.coverage_enabled = False
        self.cov = None
    
    def setup_coverage(self):
        """Set up code coverage tracking"""
        try:
            self.cov = coverage.Coverage(
                source=['intelligent_automation_engine'],
                omit=[
                    '*/tests/*',
                    '*/test_*',
                    '*/__pycache__/*',
                    '*/venv/*',
                    '*/env/*'
                ]
            )
            self.cov.start()
            self.coverage_enabled = True
            print("✓ Code coverage tracking enabled")
        except ImportError:
            print("⚠ Coverage module not available. Install with: pip install coverage")
            self.coverage_enabled = False
    
    def stop_coverage(self):
        """Stop coverage tracking and generate report"""
        if self.coverage_enabled and self.cov:
            self.cov.stop()
            self.cov.save()
            
            print("\n" + "="*60)
            print("CODE COVERAGE REPORT")
            print("="*60)
            
            # Generate console report
            self.cov.report()
            
            # Generate HTML report
            try:
                html_dir = os.path.join(project_root, 'htmlcov')
                self.cov.html_report(directory=html_dir)
                print(f"\n📊 HTML coverage report generated: {html_dir}/index.html")
            except Exception as e:
                print(f"⚠ Could not generate HTML report: {e}")
    
    def run_parallel_tests(self, verbosity: int = 2) -> unittest.TestResult:
        """Run parallel execution tests"""
        print("\n🔄 Running Parallel Execution Tests...")
        print("-" * 50)
        
        suite = create_parallel_suite()
        runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
        return runner.run(suite)
    
    def run_scheduling_tests(self, verbosity: int = 2) -> unittest.TestResult:
        """Run scheduling tests"""
        print("\n⏰ Running Scheduling Tests...")
        print("-" * 50)
        
        suite = create_scheduling_suite()
        runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
        return runner.run(suite)
    
    def run_integration_tests(self, verbosity: int = 2) -> unittest.TestResult:
        """Run integration tests"""
        print("\n🔗 Running Integration Tests...")
        print("-" * 50)
        
        suite = create_integration_test_suite()
        runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
        return runner.run(suite)
    
    def run_all_tests(self, verbosity: int = 2) -> List[unittest.TestResult]:
        """Run all test suites"""
        print("🚀 Running All Tests for Intelligent Automation Engine")
        print("=" * 60)
        
        results = []
        
        # Run each test suite
        results.append(self.run_parallel_tests(verbosity))
        results.append(self.run_scheduling_tests(verbosity))
        results.append(self.run_integration_tests(verbosity))
        
        return results
    
    def run_quick_tests(self, verbosity: int = 1) -> List[unittest.TestResult]:
        """Run a quick subset of tests"""
        print("⚡ Running Quick Test Suite...")
        print("-" * 40)
        
        # Create a smaller test suite with key tests
        suite = unittest.TestSuite()
        
        # Add a few key tests from each module
        from tests.test_parallel_execution import TestExecutionPlanner, TestDependencyAnalyzer
        from tests.test_scheduling import TestCronScheduler, TestNaturalLanguageScheduler
        from tests.test_integration import TestFullSystemIntegration
        
        # Add specific test methods
        suite.addTest(TestExecutionPlanner('test_plan_creation'))
        suite.addTest(TestDependencyAnalyzer('test_dependency_analysis'))
        suite.addTest(TestCronScheduler('test_cron_expression_parsing'))
        suite.addTest(TestNaturalLanguageScheduler('test_natural_language_parsing'))
        suite.addTest(TestFullSystemIntegration('test_parallel_execution_with_scheduling'))
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        return [result]
    
    def print_summary(self, results: List[unittest.TestResult]):
        """Print comprehensive test summary"""
        print("\n" + "="*70)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*70)
        
        total_tests = sum(result.testsRun for result in results)
        total_failures = sum(len(result.failures) for result in results)
        total_errors = sum(len(result.errors) for result in results)
        total_skipped = sum(len(result.skipped) if hasattr(result, 'skipped') else 0 for result in results)
        
        success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Total Tests Run: {total_tests}")
        print(f"✅ Passed: {total_tests - total_failures - total_errors}")
        print(f"❌ Failed: {total_failures}")
        print(f"💥 Errors: {total_errors}")
        print(f"⏭️  Skipped: {total_skipped}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Status indicator
        if total_failures == 0 and total_errors == 0:
            print("\n🎉 ALL TESTS PASSED! 🎉")
            status = "PASSED"
        elif success_rate >= 90:
            print("\n✅ MOSTLY SUCCESSFUL")
            status = "MOSTLY_PASSED"
        elif success_rate >= 70:
            print("\n⚠️  SOME ISSUES DETECTED")
            status = "PARTIAL_PASS"
        else:
            print("\n❌ SIGNIFICANT ISSUES DETECTED")
            status = "FAILED"
        
        # Detailed failure/error reporting
        if total_failures > 0 or total_errors > 0:
            print(f"\n{'='*70}")
            print("DETAILED ISSUE REPORT")
            print("="*70)
            
            for i, result in enumerate(results):
                suite_names = ["Parallel Execution", "Scheduling", "Integration"]
                suite_name = suite_names[i] if i < len(suite_names) else f"Suite {i+1}"
                
                if result.failures:
                    print(f"\n❌ {suite_name} Failures:")
                    for test, traceback in result.failures:
                        error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError:' in traceback else "Unknown assertion error"
                        print(f"  • {test}: {error_msg}")
                
                if result.errors:
                    print(f"\n💥 {suite_name} Errors:")
                    for test, traceback in result.errors:
                        error_msg = traceback.split('\n')[-2] if '\n' in traceback else traceback
                        print(f"  • {test}: {error_msg}")
        
        print("="*70)
        return status
    
    def run_performance_benchmark(self):
        """Run performance benchmarks"""
        print("\n🏃 Running Performance Benchmarks...")
        print("-" * 50)
        
        # Import performance test classes
        from tests.test_parallel_execution import TestParallelExecutionPerformance
        from tests.test_scheduling import TestSchedulingPerformance
        
        # Create performance test suite
        perf_suite = unittest.TestSuite()
        perf_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestParallelExecutionPerformance))
        perf_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSchedulingPerformance))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(perf_suite)
        
        return result


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Intelligent Automation Engine Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --quick            # Run quick test suite
  python run_tests.py --parallel         # Run only parallel tests
  python run_tests.py --scheduling       # Run only scheduling tests
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --performance      # Run performance benchmarks
  python run_tests.py --coverage         # Run with coverage tracking
  python run_tests.py --verbose          # Run with high verbosity
        """
    )
    
    parser.add_argument('--parallel', action='store_true', help='Run only parallel execution tests')
    parser.add_argument('--scheduling', action='store_true', help='Run only scheduling tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--performance', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--quick', action='store_true', help='Run quick test suite')
    parser.add_argument('--coverage', action='store_true', help='Enable code coverage tracking')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
    
    args = parser.parse_args()
    
    # Determine verbosity level
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Create test runner
    runner = TestRunner()
    
    # Setup coverage if requested
    if args.coverage:
        runner.setup_coverage()
    
    try:
        start_time = time.time()
        
        # Run specific test suites based on arguments
        if args.quick:
            results = runner.run_quick_tests(verbosity)
        elif args.parallel:
            results = [runner.run_parallel_tests(verbosity)]
        elif args.scheduling:
            results = [runner.run_scheduling_tests(verbosity)]
        elif args.integration:
            results = [runner.run_integration_tests(verbosity)]
        elif args.performance:
            results = [runner.run_performance_benchmark()]
        else:
            # Run all tests
            results = runner.run_all_tests(verbosity)
        
        end_time = time.time()
        
        # Print summary
        status = runner.print_summary(results)
        
        print(f"\n⏱️  Total execution time: {end_time - start_time:.2f} seconds")
        
        # Stop coverage tracking
        if args.coverage:
            runner.stop_coverage()
        
        # Exit with appropriate code
        if status in ["PASSED", "MOSTLY_PASSED"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during test execution: {e}")
        sys.exit(1)
    finally:
        # Ensure coverage is stopped
        if args.coverage and runner.coverage_enabled:
            runner.stop_coverage()


if __name__ == '__main__':
    main()