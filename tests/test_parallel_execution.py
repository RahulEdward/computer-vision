"""
Comprehensive Integration Tests for Parallel Execution Components

This module provides comprehensive integration tests for all parallel execution
components including ExecutionPlanner, DependencyAnalyzer, TaskScheduler,
ResourceBalancer, and SynchronizationManager.
"""

import unittest
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import tempfile
import os
import json

# Import components to test
from intelligent_automation_engine.parallel import (
    ExecutionPlanner,
    DependencyAnalyzer,
    TaskScheduler,
    ResourceBalancer,
    SynchronizationManager,
    # Data structures
    Task,
    TaskDependency,
    ExecutionPlan,
    ExecutionStrategy,
    TaskState,
    DependencyType,
    SchedulingAlgorithm,
    ScheduledTask,
    WorkerNode,
    BalancingStrategy,
    ResourceNode,
    SynchronizationType,
    SynchronizationPrimitive
)


class MockTask:
    """Mock task for testing"""
    
    def __init__(self, task_id: str, duration: float = 1.0, dependencies: List[str] = None):
        self.id = task_id
        self.name = f"Task {task_id}"
        self.duration = duration
        self.dependencies = dependencies or []
        self.executed = False
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
    
    def execute(self) -> Any:
        """Execute the mock task"""
        self.start_time = datetime.now()
        time.sleep(self.duration)
        self.executed = True
        self.end_time = datetime.now()
        self.result = f"Result from {self.name}"
        return self.result
    
    async def execute_async(self) -> Any:
        """Execute the mock task asynchronously"""
        self.start_time = datetime.now()
        await asyncio.sleep(self.duration)
        self.executed = True
        self.end_time = datetime.now()
        self.result = f"Async result from {self.name}"
        return self.result


class TestExecutionPlanner(unittest.TestCase):
    """Test cases for ExecutionPlanner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.planner = ExecutionPlanner()
        self.tasks = [
            MockTask("A", 1.0),
            MockTask("B", 1.5, ["A"]),
            MockTask("C", 2.0, ["A"]),
            MockTask("D", 1.0, ["B", "C"]),
            MockTask("E", 0.5)
        ]
    
    def test_plan_creation(self):
        """Test execution plan creation"""
        # Convert mock tasks to Task objects
        task_objects = []
        for mock_task in self.tasks:
            task = Task(
                id=mock_task.id,
                name=mock_task.name,
                function=mock_task.execute,
                estimated_duration=timedelta(seconds=mock_task.duration)
            )
            task_objects.append(task)
        
        # Add dependencies
        dependencies = []
        for mock_task in self.tasks:
            for dep_id in mock_task.dependencies:
                dep = TaskDependency(
                    source_task_id=dep_id,
                    target_task_id=mock_task.id,
                    dependency_type="finish_to_start"
                )
                dependencies.append(dep)
        
        # Create execution plan
        plan = self.planner.create_execution_plan(
            tasks=task_objects,
            dependencies=dependencies,
            strategy=ExecutionStrategy.PARALLEL_OPTIMIZED
        )
        
        # Verify plan creation
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertEqual(len(plan.tasks), len(task_objects))
        self.assertEqual(len(plan.dependencies), len(dependencies))
        self.assertGreater(len(plan.execution_stages), 0)
    
    def test_dependency_analysis(self):
        """Test dependency analysis"""
        # Create tasks with complex dependencies
        task_objects = []
        dependencies = []
        
        for mock_task in self.tasks:
            task = Task(
                id=mock_task.id,
                name=mock_task.name,
                function=mock_task.execute
            )
            task_objects.append(task)
            
            for dep_id in mock_task.dependencies:
                dep = TaskDependency(
                    source_task_id=dep_id,
                    target_task_id=mock_task.id,
                    dependency_type="finish_to_start"
                )
                dependencies.append(dep)
        
        # Analyze dependencies
        analysis = self.planner.analyze_dependencies(task_objects, dependencies)
        
        # Verify analysis results
        self.assertIn('dependency_graph', analysis)
        self.assertIn('critical_path', analysis)
        self.assertIn('parallel_opportunities', analysis)
        self.assertIn('bottlenecks', analysis)
    
    def test_parallel_execution(self):
        """Test parallel execution of independent tasks"""
        # Create independent tasks
        independent_tasks = [
            Task(id="T1", name="Task 1", function=lambda: time.sleep(0.1)),
            Task(id="T2", name="Task 2", function=lambda: time.sleep(0.1)),
            Task(id="T3", name="Task 3", function=lambda: time.sleep(0.1))
        ]
        
        # Create execution plan
        plan = self.planner.create_execution_plan(
            tasks=independent_tasks,
            dependencies=[],
            strategy=ExecutionStrategy.PARALLEL_OPTIMIZED
        )
        
        # Execute plan
        start_time = time.time()
        result = self.planner.execute_plan(plan)
        end_time = time.time()
        
        # Verify parallel execution (should be faster than sequential)
        execution_time = end_time - start_time
        self.assertLess(execution_time, 0.25)  # Should be much less than 0.3 seconds
        self.assertTrue(result.success)


class TestDependencyAnalyzer(unittest.TestCase):
    """Test cases for DependencyAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = DependencyAnalyzer()
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        # Create tasks with circular dependency
        tasks = [
            Task(id="A", name="Task A"),
            Task(id="B", name="Task B"),
            Task(id="C", name="Task C")
        ]
        
        dependencies = [
            TaskDependency("A", "B", "finish_to_start"),
            TaskDependency("B", "C", "finish_to_start"),
            TaskDependency("C", "A", "finish_to_start")  # Circular
        ]
        
        # Analyze dependencies
        result = self.analyzer.analyze_dependencies(tasks, dependencies)
        
        # Verify circular dependency detection
        self.assertTrue(result.has_circular_dependencies)
        self.assertGreater(len(result.circular_dependency_chains), 0)
    
    def test_critical_path_calculation(self):
        """Test critical path calculation"""
        # Create tasks with known critical path
        tasks = [
            Task(id="A", name="Task A", estimated_duration=timedelta(hours=2)),
            Task(id="B", name="Task B", estimated_duration=timedelta(hours=3)),
            Task(id="C", name="Task C", estimated_duration=timedelta(hours=1)),
            Task(id="D", name="Task D", estimated_duration=timedelta(hours=2))
        ]
        
        dependencies = [
            TaskDependency("B", "A", "finish_to_start"),
            TaskDependency("C", "A", "finish_to_start"),
            TaskDependency("D", "B", "finish_to_start"),
            TaskDependency("D", "C", "finish_to_start")
        ]
        
        # Analyze dependencies
        result = self.analyzer.analyze_dependencies(tasks, dependencies)
        
        # Verify critical path
        self.assertIsNotNone(result.critical_path)
        self.assertGreater(len(result.critical_path), 0)
        self.assertGreater(result.total_duration.total_seconds(), 0)


class TestTaskScheduler(unittest.TestCase):
    """Test cases for TaskScheduler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = TaskScheduler()
        
        # Add worker nodes
        for i in range(3):
            worker = WorkerNode(
                id=f"worker_{i}",
                name=f"Worker {i}",
                max_concurrent_tasks=2
            )
            self.scheduler.add_worker(worker)
    
    def test_task_scheduling(self):
        """Test basic task scheduling"""
        # Create scheduled tasks
        tasks = []
        for i in range(5):
            task = ScheduledTask(
                id=f"task_{i}",
                name=f"Task {i}",
                function=lambda: time.sleep(0.1),
                estimated_duration=timedelta(seconds=0.1)
            )
            tasks.append(task)
        
        # Schedule tasks
        for task in tasks:
            success = self.scheduler.schedule_task(task)
            self.assertTrue(success)
        
        # Start scheduler
        self.scheduler.start()
        
        # Wait for tasks to complete
        time.sleep(1.0)
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Verify task completion
        metrics = self.scheduler.get_scheduling_metrics()
        self.assertGreater(metrics.total_tasks_scheduled, 0)
    
    def test_scheduling_algorithms(self):
        """Test different scheduling algorithms"""
        algorithms = [
            SchedulingAlgorithm.FIFO,
            SchedulingAlgorithm.PRIORITY,
            SchedulingAlgorithm.SHORTEST_JOB_FIRST
        ]
        
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                # Create scheduler with specific algorithm
                scheduler = TaskScheduler(default_algorithm=algorithm)
                
                # Add workers
                worker = WorkerNode(id="test_worker", name="Test Worker")
                scheduler.add_worker(worker)
                
                # Schedule tasks
                for i in range(3):
                    task = ScheduledTask(
                        id=f"task_{i}",
                        name=f"Task {i}",
                        function=lambda: None,
                        priority=i
                    )
                    scheduler.schedule_task(task)
                
                # Verify scheduling
                queue_status = scheduler.get_queue_status()
                self.assertGreater(queue_status.total_tasks, 0)


class TestResourceBalancer(unittest.TestCase):
    """Test cases for ResourceBalancer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.balancer = ResourceBalancer()
        
        # Add resource nodes
        for i in range(3):
            node = ResourceNode(
                id=f"node_{i}",
                name=f"Node {i}",
                cpu_cores=4,
                memory_gb=8,
                max_concurrent_tasks=5
            )
            self.balancer.add_node(node)
    
    def test_load_balancing(self):
        """Test load balancing functionality"""
        # Start monitoring
        self.balancer.start_monitoring()
        
        # Simulate load on nodes
        for i, node_id in enumerate([f"node_{j}" for j in range(3)]):
            load = 0.3 + (i * 0.2)  # Different loads
            self.balancer.update_node_load(node_id, cpu_usage=load, memory_usage=load)
        
        # Test node selection
        selected_node = self.balancer.select_node(BalancingStrategy.LEAST_LOADED)
        self.assertIsNotNone(selected_node)
        
        # Get load distribution
        distribution = self.balancer.get_load_distribution()
        self.assertGreater(len(distribution), 0)
        
        # Stop monitoring
        self.balancer.stop_monitoring()
    
    def test_balancing_strategies(self):
        """Test different balancing strategies"""
        strategies = [
            BalancingStrategy.ROUND_ROBIN,
            BalancingStrategy.LEAST_LOADED,
            BalancingStrategy.WEIGHTED_ROUND_ROBIN,
            BalancingStrategy.RESOURCE_AWARE
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                selected_node = self.balancer.select_node(strategy)
                self.assertIsNotNone(selected_node)


class TestSynchronizationManager(unittest.TestCase):
    """Test cases for SynchronizationManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sync_manager = SynchronizationManager()
        self.sync_manager.start_deadlock_detection()
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.sync_manager.stop_deadlock_detection()
    
    def test_mutex_synchronization(self):
        """Test mutex synchronization"""
        # Create mutex
        mutex_id = self.sync_manager.create_mutex("test_mutex")
        self.assertIsNotNone(mutex_id)
        
        # Test acquisition and release
        success = self.sync_manager.acquire_primitive(mutex_id, "task_1")
        self.assertTrue(success)
        
        # Test that second acquisition fails
        success = self.sync_manager.acquire_primitive(mutex_id, "task_2", timeout=0.1)
        self.assertFalse(success)
        
        # Release and test second acquisition
        self.sync_manager.release_primitive(mutex_id, "task_1")
        success = self.sync_manager.acquire_primitive(mutex_id, "task_2")
        self.assertTrue(success)
        
        # Clean up
        self.sync_manager.release_primitive(mutex_id, "task_2")
    
    def test_semaphore_synchronization(self):
        """Test semaphore synchronization"""
        # Create semaphore with capacity 2
        semaphore_id = self.sync_manager.create_semaphore("test_semaphore", initial_count=2)
        self.assertIsNotNone(semaphore_id)
        
        # Test multiple acquisitions
        success1 = self.sync_manager.acquire_primitive(semaphore_id, "task_1")
        success2 = self.sync_manager.acquire_primitive(semaphore_id, "task_2")
        self.assertTrue(success1)
        self.assertTrue(success2)
        
        # Test that third acquisition fails
        success3 = self.sync_manager.acquire_primitive(semaphore_id, "task_3", timeout=0.1)
        self.assertFalse(success3)
        
        # Release one and test third acquisition
        self.sync_manager.release_primitive(semaphore_id, "task_1")
        success3 = self.sync_manager.acquire_primitive(semaphore_id, "task_3")
        self.assertTrue(success3)
        
        # Clean up
        self.sync_manager.release_primitive(semaphore_id, "task_2")
        self.sync_manager.release_primitive(semaphore_id, "task_3")
    
    def test_barrier_synchronization(self):
        """Test barrier synchronization"""
        # Create barrier for 3 tasks
        barrier_id = self.sync_manager.create_barrier("test_barrier", party_count=3)
        self.assertIsNotNone(barrier_id)
        
        # Test barrier waiting (this is a simplified test)
        # In a real scenario, this would involve multiple threads
        results = []
        
        def wait_at_barrier(task_id):
            result = self.sync_manager.wait_at_barrier(barrier_id, task_id, timeout=1.0)
            results.append(result)
        
        # Simulate multiple tasks waiting
        threads = []
        for i in range(3):
            thread = threading.Thread(target=wait_at_barrier, args=[f"task_{i}"])
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All should succeed when barrier is reached
        self.assertEqual(len(results), 3)
    
    def test_deadlock_detection(self):
        """Test deadlock detection"""
        # Create two mutexes
        mutex1_id = self.sync_manager.create_mutex("mutex_1")
        mutex2_id = self.sync_manager.create_mutex("mutex_2")
        
        # Acquire mutex1 with task1
        self.sync_manager.acquire_primitive(mutex1_id, "task_1")
        
        # Acquire mutex2 with task2
        self.sync_manager.acquire_primitive(mutex2_id, "task_2")
        
        # Try to create deadlock scenario
        # task_1 tries to acquire mutex2 (will block)
        # task_2 tries to acquire mutex1 (will block)
        # This should be detected as a potential deadlock
        
        # Get synchronization metrics
        metrics = self.sync_manager.get_synchronization_metrics()
        self.assertIsNotNone(metrics)
        
        # Clean up
        self.sync_manager.release_primitive(mutex1_id, "task_1")
        self.sync_manager.release_primitive(mutex2_id, "task_2")


class TestIntegratedParallelExecution(unittest.TestCase):
    """Integration tests for all parallel execution components working together"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.planner = ExecutionPlanner()
        self.scheduler = TaskScheduler()
        self.balancer = ResourceBalancer()
        self.sync_manager = SynchronizationManager()
        
        # Set up workers and resources
        for i in range(2):
            worker = WorkerNode(id=f"worker_{i}", name=f"Worker {i}")
            self.scheduler.add_worker(worker)
            
            node = ResourceNode(id=f"node_{i}", name=f"Node {i}")
            self.balancer.add_node(node)
    
    def test_end_to_end_execution(self):
        """Test end-to-end parallel execution workflow"""
        # Create a complex workflow with dependencies
        tasks = []
        dependencies = []
        
        # Task A (independent)
        task_a = Task(
            id="A",
            name="Task A",
            function=lambda: time.sleep(0.1),
            estimated_duration=timedelta(seconds=0.1)
        )
        tasks.append(task_a)
        
        # Task B (depends on A)
        task_b = Task(
            id="B",
            name="Task B",
            function=lambda: time.sleep(0.1),
            estimated_duration=timedelta(seconds=0.1)
        )
        tasks.append(task_b)
        dependencies.append(TaskDependency("B", "A", "finish_to_start"))
        
        # Task C (independent)
        task_c = Task(
            id="C",
            name="Task C",
            function=lambda: time.sleep(0.1),
            estimated_duration=timedelta(seconds=0.1)
        )
        tasks.append(task_c)
        
        # Task D (depends on B and C)
        task_d = Task(
            id="D",
            name="Task D",
            function=lambda: time.sleep(0.1),
            estimated_duration=timedelta(seconds=0.1)
        )
        tasks.append(task_d)
        dependencies.append(TaskDependency("D", "B", "finish_to_start"))
        dependencies.append(TaskDependency("D", "C", "finish_to_start"))
        
        # Create execution plan
        plan = self.planner.create_execution_plan(
            tasks=tasks,
            dependencies=dependencies,
            strategy=ExecutionStrategy.PARALLEL_OPTIMIZED
        )
        
        # Verify plan creation
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertTrue(plan.is_valid)
        
        # Execute plan
        start_time = time.time()
        result = self.planner.execute_plan(plan)
        end_time = time.time()
        
        # Verify execution
        self.assertTrue(result.success)
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)  # Should complete quickly with parallelization
    
    def test_resource_aware_scheduling(self):
        """Test resource-aware scheduling integration"""
        # Start resource monitoring
        self.balancer.start_monitoring()
        
        # Create tasks with different resource requirements
        tasks = []
        for i in range(5):
            task = Task(
                id=f"task_{i}",
                name=f"Task {i}",
                function=lambda: time.sleep(0.1),
                resource_requirements={'cpu': 0.5, 'memory': 1.0}
            )
            tasks.append(task)
        
        # Create execution plan with resource awareness
        plan = self.planner.create_execution_plan(
            tasks=tasks,
            dependencies=[],
            strategy=ExecutionStrategy.RESOURCE_OPTIMIZED
        )
        
        # Execute with resource balancing
        result = self.planner.execute_plan(plan)
        
        # Verify execution
        self.assertTrue(result.success)
        
        # Stop monitoring
        self.balancer.stop_monitoring()
    
    def test_synchronized_execution(self):
        """Test synchronized execution with synchronization primitives"""
        # Create shared resource mutex
        mutex_id = self.sync_manager.create_mutex("shared_resource")
        
        # Create tasks that need synchronization
        shared_counter = {'value': 0}
        
        def synchronized_task(task_id: str):
            # Acquire mutex
            if self.sync_manager.acquire_primitive(mutex_id, task_id):
                try:
                    # Critical section
                    current = shared_counter['value']
                    time.sleep(0.01)  # Simulate work
                    shared_counter['value'] = current + 1
                finally:
                    # Release mutex
                    self.sync_manager.release_primitive(mutex_id, task_id)
                return True
            return False
        
        # Create tasks
        tasks = []
        for i in range(5):
            task = Task(
                id=f"sync_task_{i}",
                name=f"Synchronized Task {i}",
                function=lambda tid=f"sync_task_{i}": synchronized_task(tid)
            )
            tasks.append(task)
        
        # Execute tasks
        plan = self.planner.create_execution_plan(
            tasks=tasks,
            dependencies=[],
            strategy=ExecutionStrategy.PARALLEL_OPTIMIZED
        )
        
        result = self.planner.execute_plan(plan)
        
        # Verify synchronized execution
        self.assertTrue(result.success)
        self.assertEqual(shared_counter['value'], 5)  # All tasks should have incremented


class TestPerformanceAndScalability(unittest.TestCase):
    """Performance and scalability tests"""
    
    def test_large_task_set_performance(self):
        """Test performance with large number of tasks"""
        planner = ExecutionPlanner()
        
        # Create large number of independent tasks
        num_tasks = 100
        tasks = []
        for i in range(num_tasks):
            task = Task(
                id=f"task_{i}",
                name=f"Task {i}",
                function=lambda: None  # No-op for performance testing
            )
            tasks.append(task)
        
        # Measure plan creation time
        start_time = time.time()
        plan = planner.create_execution_plan(
            tasks=tasks,
            dependencies=[],
            strategy=ExecutionStrategy.PARALLEL_OPTIMIZED
        )
        plan_time = time.time() - start_time
        
        # Verify reasonable performance
        self.assertLess(plan_time, 5.0)  # Should create plan in under 5 seconds
        self.assertEqual(len(plan.tasks), num_tasks)
    
    def test_complex_dependency_performance(self):
        """Test performance with complex dependency graphs"""
        planner = ExecutionPlanner()
        
        # Create tasks with complex dependencies
        num_tasks = 50
        tasks = []
        dependencies = []
        
        for i in range(num_tasks):
            task = Task(id=f"task_{i}", name=f"Task {i}", function=lambda: None)
            tasks.append(task)
            
            # Create dependencies (each task depends on previous 2-3 tasks)
            for j in range(max(0, i-3), i):
                dep = TaskDependency(f"task_{i}", f"task_{j}", "finish_to_start")
                dependencies.append(dep)
        
        # Measure analysis time
        start_time = time.time()
        plan = planner.create_execution_plan(
            tasks=tasks,
            dependencies=dependencies,
            strategy=ExecutionStrategy.DEPENDENCY_OPTIMIZED
        )
        analysis_time = time.time() - start_time
        
        # Verify reasonable performance
        self.assertLess(analysis_time, 10.0)  # Should analyze in under 10 seconds
        self.assertTrue(plan.is_valid)


def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestExecutionPlanner,
        TestDependencyAnalyzer,
        TestTaskScheduler,
        TestResourceBalancer,
        TestSynchronizationManager,
        TestIntegratedParallelExecution,
        TestPerformanceAndScalability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == '__main__':
    # Run comprehensive test suite
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PARALLEL EXECUTION INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            newline = '\n'
            error_msg = traceback.split('AssertionError: ')[-1].split(newline)[0]
            print(f"  - {test}: {error_msg}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            newline = '\n'
            error_msg = traceback.split(newline)[-2]
            print(f"  - {test}: {error_msg}")
    
    print(f"{'='*60}")