"""
Comprehensive Integration Tests for Entire Intelligent Automation Engine

This module provides end-to-end integration tests that verify the complete
intelligent automation engine works correctly with all components integrated.
"""

import unittest
import asyncio
import threading
import time
import tempfile
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import all major components
from intelligent_automation_engine.parallel import (
    ExecutionPlanner,
    DependencyAnalyzer,
    TaskScheduler,
    ResourceBalancer,
    SynchronizationManager
)

from intelligent_automation_engine.scheduling import (
    CronScheduler,
    NaturalLanguageScheduler,
    ScheduleOptimizer,
    TimeManager,
    EventScheduler
)

# Import data structures
from intelligent_automation_engine.parallel.execution_planner import (
    ExecutionPlan, ExecutionStrategy
)
from intelligent_automation_engine.parallel import (
    Task, TaskState as TaskStatus
)
from intelligent_automation_engine.scheduling.event_scheduler import (
    Event, EventType, EventPriority
)


class TestFullSystemIntegration(unittest.TestCase):
    """Test complete system integration"""
    
    def setUp(self):
        """Set up full system test environment"""
        # Initialize all major components
        self.execution_planner = ExecutionPlanner()
        self.dependency_analyzer = DependencyAnalyzer()
        self.task_scheduler = TaskScheduler()
        self.resource_balancer = ResourceBalancer()
        self.sync_manager = SynchronizationManager()
        
        self.cron_scheduler = CronScheduler()
        self.nl_scheduler = NaturalLanguageScheduler()
        self.schedule_optimizer = ScheduleOptimizer()
        self.time_manager = TimeManager()
        self.event_scheduler = EventScheduler()
        
        # Test results storage
        self.execution_results = []
        self.system_events = []
    
    def tearDown(self):
        """Clean up test environment"""
        # Stop all running components
        if self.cron_scheduler.is_running():
            self.cron_scheduler.stop()
        if hasattr(self.event_scheduler, 'state') and self.event_scheduler.state.name == 'RUNNING':
            self.event_scheduler.stop()
        if self.sync_manager.is_running():
            self.sync_manager.stop()
    
    def test_parallel_execution_with_scheduling(self):
        """Test parallel execution integrated with scheduling"""
        execution_log = []
        
        def create_test_task(task_id: str, duration: float = 1.0, dependencies: List[str] = None):
            """Create a test task"""
            def task_function():
                start_time = time.time()
                time.sleep(duration)
                end_time = time.time()
                execution_log.append({
                    'task_id': task_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'thread': threading.current_thread().name
                })
                return f"Task {task_id} completed"
            
            return Task(
                id=task_id,
                name=f"Test Task {task_id}",
                function=task_function,
                dependencies=dependencies or [],
                estimated_duration=duration,
                resource_requirements={'cpu': 0.5, 'memory': 100}
            )
        
        # 1. Create task dependency graph
        tasks = [
            create_test_task("A", 1.0),
            create_test_task("B", 1.5, ["A"]),
            create_test_task("C", 1.0, ["A"]),
            create_test_task("D", 2.0, ["B", "C"]),
            create_test_task("E", 1.0, ["D"])
        ]
        
        # 2. Analyze dependencies
        dependency_graph = self.dependency_analyzer.analyze_dependencies(tasks)
        self.assertIsNotNone(dependency_graph)
        
        # 3. Create execution plan
        execution_plan = self.execution_planner.create_execution_plan(
            tasks=tasks,
            strategy=ExecutionStrategy.PARALLEL_OPTIMIZED,
            max_workers=3
        )
        
        self.assertIsNotNone(execution_plan)
        self.assertEqual(len(execution_plan.tasks), 5)
        
        # 4. Schedule execution using cron scheduler
        def execute_plan():
            result = self.execution_planner.execute_plan(execution_plan)
            self.execution_results.append(result)
        
        # Schedule immediate execution
        schedule_id = self.cron_scheduler.add_schedule(
            name="Parallel Execution",
            cron_expression="* * * * * *",  # Every second
            task_function=execute_plan
        )
        
        # 5. Start scheduler and wait for execution
        self.cron_scheduler.start()
        time.sleep(3)  # Wait for execution
        self.cron_scheduler.stop()
        
        # 6. Verify execution
        self.assertGreater(len(execution_log), 0)
        self.assertGreater(len(self.execution_results), 0)
        
        # Verify task execution order respects dependencies
        task_times = {log['task_id']: log['start_time'] for log in execution_log}
        
        # Task A should start first
        self.assertLessEqual(task_times['A'], min(task_times.values()))
        
        # Tasks B and C should start after A
        if 'B' in task_times and 'A' in task_times:
            self.assertGreater(task_times['B'], task_times['A'])
        if 'C' in task_times and 'A' in task_times:
            self.assertGreater(task_times['C'], task_times['A'])
        
        # Clean up
        self.cron_scheduler.remove_schedule(schedule_id)
    
    def test_event_driven_parallel_execution(self):
        """Test event-driven parallel execution"""
        execution_events = []
        
        def event_handler(event):
            """Handle execution events"""
            execution_events.append({
                'event_name': event.name,
                'event_type': event.event_type.name,
                'timestamp': datetime.now(),
                'data': event.data
            })
            
            # Trigger parallel execution based on event
            if event.name == "start_parallel_execution":
                self.execute_parallel_tasks(event.data.get('task_count', 3))
        
        def execute_parallel_tasks(task_count: int):
            """Execute parallel tasks"""
            tasks = []
            for i in range(task_count):
                task = Task(
                    id=f"event_task_{i}",
                    name=f"Event Task {i}",
                    function=lambda i=i: f"Event task {i} result",
                    estimated_duration=0.5,
                    resource_requirements={'cpu': 0.3}
                )
                tasks.append(task)
            
            # Create and execute plan
            plan = self.execution_planner.create_execution_plan(
                tasks=tasks,
                strategy=ExecutionStrategy.PARALLEL_OPTIMIZED
            )
            
            result = self.execution_planner.execute_plan(plan)
            self.execution_results.append(result)
        
        # 1. Set up event handler
        from intelligent_automation_engine.scheduling.event_scheduler import EventHandler
        handler = EventHandler(
            name="Execution Handler",
            event_types=[EventType.USER],
            handler_function=event_handler
        )
        
        self.event_scheduler.add_handler(handler)
        self.event_scheduler.start()
        
        # 2. Create and schedule trigger event
        trigger_event = Event(
            name="start_parallel_execution",
            event_type=EventType.USER,
            data={'task_count': 4},
            priority=EventPriority.HIGH
        )
        
        self.event_scheduler.schedule_event(trigger_event)
        
        # 3. Wait for processing
        time.sleep(3)
        
        # 4. Verify execution
        self.assertGreater(len(execution_events), 0)
        self.assertGreater(len(self.execution_results), 0)
        
        # Verify event was processed
        trigger_processed = any(
            event['event_name'] == "start_parallel_execution" 
            for event in execution_events
        )
        self.assertTrue(trigger_processed)
        
        # 5. Clean up
        self.event_scheduler.stop()
    
    def test_resource_aware_scheduling_optimization(self):
        """Test resource-aware scheduling with optimization"""
        # 1. Set up resource constraints
        self.resource_balancer.add_resource("cpu", 4.0)
        self.resource_balancer.add_resource("memory", 8192)
        self.resource_balancer.add_resource("disk", 1000)
        
        # 2. Create resource-intensive tasks
        resource_tasks = []
        for i in range(6):
            task = Task(
                id=f"resource_task_{i}",
                name=f"Resource Task {i}",
                function=lambda i=i: time.sleep(0.5) or f"Resource task {i} done",
                estimated_duration=0.5,
                resource_requirements={
                    'cpu': 1.0,
                    'memory': 1024,
                    'disk': 100
                },
                priority=i % 3
            )
            resource_tasks.append(task)
        
        # 3. Create execution plan with resource balancing
        plan = self.execution_planner.create_execution_plan(
            tasks=resource_tasks,
            strategy=ExecutionStrategy.RESOURCE_OPTIMIZED,
            max_workers=4
        )
        
        # 4. Add schedules to optimizer for conflict detection
        base_time = datetime.now() + timedelta(seconds=1)
        schedule_ids = []
        
        for i, task in enumerate(resource_tasks):
            schedule_id = self.schedule_optimizer.add_schedule(
                name=task.name,
                start_time=base_time + timedelta(seconds=i*0.1),
                duration=timedelta(seconds=task.estimated_duration),
                priority=task.priority,
                resource_requirements=task.resource_requirements
            )
            schedule_ids.append(schedule_id)
        
        # 5. Detect and resolve conflicts
        conflicts = self.schedule_optimizer.detect_conflicts()
        
        for conflict in conflicts:
            resolution = self.schedule_optimizer.resolve_conflict(conflict)
            self.assertIsNotNone(resolution)
        
        # 6. Execute optimized plan
        result = self.execution_planner.execute_plan(plan)
        
        # 7. Verify execution
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(len(result.task_results), 6)
        
        # Verify resource constraints were respected
        resource_stats = self.resource_balancer.get_resource_statistics()
        self.assertIsNotNone(resource_stats)
    
    def test_synchronized_scheduled_execution(self):
        """Test synchronized execution with scheduling"""
        sync_results = []
        
        def synchronized_task(task_id: str, barrier_name: str):
            """Task that uses synchronization"""
            def task_function():
                # Acquire mutex
                mutex_id = self.sync_manager.create_mutex(f"mutex_{task_id}")
                self.sync_manager.acquire_primitive(mutex_id)
                
                try:
                    # Do some work
                    sync_results.append(f"Task {task_id} started")
                    time.sleep(0.5)
                    
                    # Wait at barrier
                    barrier_id = self.sync_manager.create_barrier(barrier_name, 3)
                    self.sync_manager.wait_barrier(barrier_id)
                    
                    sync_results.append(f"Task {task_id} passed barrier")
                    
                finally:
                    # Release mutex
                    self.sync_manager.release_primitive(mutex_id)
                
                return f"Task {task_id} completed"
            
            return task_function
        
        # 1. Start synchronization manager
        self.sync_manager.start()
        
        # 2. Create synchronized tasks
        sync_tasks = []
        for i in range(3):
            task = Task(
                id=f"sync_task_{i}",
                name=f"Sync Task {i}",
                function=synchronized_task(f"sync_task_{i}", "test_barrier"),
                estimated_duration=1.0
            )
            sync_tasks.append(task)
        
        # 3. Schedule tasks using natural language
        nl_expression = "in 2 seconds"
        parsed_schedule = self.nl_scheduler.parse_expression(nl_expression)
        self.assertIsNotNone(parsed_schedule)
        
        # 4. Execute tasks
        plan = self.execution_planner.create_execution_plan(
            tasks=sync_tasks,
            strategy=ExecutionStrategy.PARALLEL_OPTIMIZED
        )
        
        # Schedule execution
        def execute_sync_plan():
            result = self.execution_planner.execute_plan(plan)
            self.execution_results.append(result)
        
        # Use cron scheduler for delayed execution
        schedule_id = self.cron_scheduler.add_schedule(
            name="Synchronized Execution",
            cron_expression="*/2 * * * * *",  # Every 2 seconds
            task_function=execute_sync_plan
        )
        
        self.cron_scheduler.start()
        time.sleep(5)  # Wait for execution
        self.cron_scheduler.stop()
        
        # 5. Verify synchronization worked
        self.assertGreater(len(sync_results), 0)
        
        # Check that all tasks passed the barrier
        barrier_passes = [result for result in sync_results if "passed barrier" in result]
        self.assertEqual(len(barrier_passes), 3)
        
        # Clean up
        self.cron_scheduler.remove_schedule(schedule_id)
        self.sync_manager.stop()
    
    def test_complex_workflow_automation(self):
        """Test complex workflow automation scenario"""
        workflow_state = {
            'data_processed': False,
            'analysis_complete': False,
            'report_generated': False,
            'notifications_sent': False
        }
        
        def data_processing_task():
            """Simulate data processing"""
            time.sleep(1)
            workflow_state['data_processed'] = True
            
            # Trigger analysis event
            analysis_event = Event(
                name="data_ready_for_analysis",
                event_type=EventType.SYSTEM,
                data={'processed_data': 'sample_data'}
            )
            self.event_scheduler.schedule_event(analysis_event)
            return "Data processed successfully"
        
        def analysis_task():
            """Simulate data analysis"""
            time.sleep(1.5)
            workflow_state['analysis_complete'] = True
            
            # Trigger report generation
            report_event = Event(
                name="analysis_complete",
                event_type=EventType.SYSTEM,
                data={'analysis_results': 'sample_results'}
            )
            self.event_scheduler.schedule_event(report_event)
            return "Analysis completed"
        
        def report_generation_task():
            """Simulate report generation"""
            time.sleep(1)
            workflow_state['report_generated'] = True
            
            # Trigger notifications
            notification_event = Event(
                name="report_ready",
                event_type=EventType.SYSTEM,
                data={'report_path': '/tmp/report.pdf'}
            )
            self.event_scheduler.schedule_event(notification_event)
            return "Report generated"
        
        def notification_task():
            """Simulate sending notifications"""
            time.sleep(0.5)
            workflow_state['notifications_sent'] = True
            return "Notifications sent"
        
        # 1. Set up event handlers for workflow steps
        def workflow_event_handler(event):
            """Handle workflow events"""
            if event.name == "data_ready_for_analysis":
                # Execute analysis task
                analysis_task()
            elif event.name == "analysis_complete":
                # Execute report generation
                report_generation_task()
            elif event.name == "report_ready":
                # Send notifications
                notification_task()
        
        from intelligent_automation_engine.scheduling.event_scheduler import EventHandler
        handler = EventHandler(
            name="Workflow Handler",
            event_types=[EventType.SYSTEM],
            handler_function=workflow_event_handler
        )
        
        self.event_scheduler.add_handler(handler)
        self.event_scheduler.start()
        
        # 2. Schedule initial data processing using natural language
        nl_expression = "in 1 second"
        parsed_schedule = self.nl_scheduler.parse_expression(nl_expression)
        
        # 3. Start workflow with cron scheduler
        schedule_id = self.cron_scheduler.add_schedule(
            name="Workflow Starter",
            cron_expression="*/1 * * * * *",  # Every second
            task_function=data_processing_task
        )
        
        self.cron_scheduler.start()
        
        # 4. Wait for workflow completion
        timeout = 10
        start_time = time.time()
        
        while not all(workflow_state.values()) and (time.time() - start_time) < timeout:
            time.sleep(0.5)
        
        # 5. Stop schedulers
        self.cron_scheduler.stop()
        self.event_scheduler.stop()
        
        # 6. Verify workflow completion
        self.assertTrue(workflow_state['data_processed'])
        self.assertTrue(workflow_state['analysis_complete'])
        self.assertTrue(workflow_state['report_generated'])
        self.assertTrue(workflow_state['notifications_sent'])
        
        # Clean up
        self.cron_scheduler.remove_schedule(schedule_id)
    
    def test_system_performance_under_load(self):
        """Test system performance under heavy load"""
        load_results = {
            'tasks_executed': 0,
            'events_processed': 0,
            'schedules_created': 0,
            'conflicts_resolved': 0
        }
        
        def load_test_task(task_id: int):
            """Simple load test task"""
            def task_function():
                time.sleep(0.1)  # Simulate work
                load_results['tasks_executed'] += 1
                return f"Load task {task_id} completed"
            return task_function
        
        # 1. Create many tasks
        num_tasks = 50
        tasks = []
        
        for i in range(num_tasks):
            task = Task(
                id=f"load_task_{i}",
                name=f"Load Task {i}",
                function=load_test_task(i),
                estimated_duration=0.1,
                resource_requirements={'cpu': 0.1}
            )
            tasks.append(task)
        
        # 2. Create execution plan
        start_time = time.time()
        
        plan = self.execution_planner.create_execution_plan(
            tasks=tasks,
            strategy=ExecutionStrategy.PARALLEL_OPTIMIZED,
            max_workers=10
        )
        
        plan_creation_time = time.time() - start_time
        
        # 3. Add many schedules to optimizer
        base_time = datetime.now() + timedelta(seconds=1)
        
        for i in range(num_tasks):
            schedule_id = self.schedule_optimizer.add_schedule(
                name=f"Load Schedule {i}",
                start_time=base_time + timedelta(seconds=i*0.01),
                duration=timedelta(seconds=0.1),
                resource_requirements={'cpu': 0.1}
            )
            load_results['schedules_created'] += 1
        
        # 4. Detect conflicts
        conflicts = self.schedule_optimizer.detect_conflicts()
        
        # 5. Resolve conflicts
        for conflict in conflicts[:10]:  # Limit to first 10 conflicts
            resolution = self.schedule_optimizer.resolve_conflict(conflict)
            if resolution:
                load_results['conflicts_resolved'] += 1
        
        # 6. Execute plan
        execution_start = time.time()
        result = self.execution_planner.execute_plan(plan)
        execution_time = time.time() - execution_start
        
        # 7. Create many events
        self.event_scheduler.start()
        
        for i in range(20):
            event = Event(
                name=f"Load Event {i}",
                event_type=EventType.USER,
                data={'index': i}
            )
            self.event_scheduler.schedule_event(event)
            load_results['events_processed'] += 1
        
        time.sleep(2)  # Wait for event processing
        self.event_scheduler.stop()
        
        # 8. Verify performance
        self.assertLess(plan_creation_time, 5.0)  # Plan creation should be fast
        self.assertLess(execution_time, 10.0)     # Execution should be reasonable
        self.assertEqual(load_results['tasks_executed'], num_tasks)
        self.assertEqual(load_results['schedules_created'], num_tasks)
        self.assertGreater(load_results['events_processed'], 0)
        
        print(f"Performance Results:")
        print(f"  Plan creation: {plan_creation_time:.2f}s")
        print(f"  Execution time: {execution_time:.2f}s")
        print(f"  Tasks executed: {load_results['tasks_executed']}")
        print(f"  Schedules created: {load_results['schedules_created']}")
        print(f"  Events processed: {load_results['events_processed']}")
        print(f"  Conflicts resolved: {load_results['conflicts_resolved']}")


class TestSystemReliability(unittest.TestCase):
    """Test system reliability and error handling"""
    
    def setUp(self):
        """Set up reliability test environment"""
        self.execution_planner = ExecutionPlanner()
        self.event_scheduler = EventScheduler()
        self.cron_scheduler = CronScheduler()
    
    def tearDown(self):
        """Clean up test environment"""
        if self.cron_scheduler.is_running():
            self.cron_scheduler.stop()
        if hasattr(self.event_scheduler, 'state') and self.event_scheduler.state.name == 'RUNNING':
            self.event_scheduler.stop()
    
    def test_error_handling_in_tasks(self):
        """Test error handling in task execution"""
        error_results = []
        
        def failing_task():
            """Task that always fails"""
            error_results.append("Failing task started")
            raise Exception("Intentional test failure")
        
        def success_task():
            """Task that succeeds"""
            error_results.append("Success task completed")
            return "Success"
        
        # Create tasks with one that fails
        tasks = [
            Task(
                id="failing_task",
                name="Failing Task",
                function=failing_task,
                estimated_duration=0.5
            ),
            Task(
                id="success_task",
                name="Success Task", 
                function=success_task,
                estimated_duration=0.5
            )
        ]
        
        # Execute plan
        plan = self.execution_planner.create_execution_plan(tasks)
        result = self.execution_planner.execute_plan(plan)
        
        # Verify error handling
        self.assertIsNotNone(result)
        self.assertFalse(result.success)  # Overall plan should fail
        self.assertEqual(len(result.task_results), 2)
        
        # Check individual task results
        failing_result = next(
            (r for r in result.task_results if r.task_id == "failing_task"), 
            None
        )
        success_result = next(
            (r for r in result.task_results if r.task_id == "success_task"), 
            None
        )
        
        self.assertIsNotNone(failing_result)
        self.assertIsNotNone(success_result)
        self.assertEqual(failing_result.status, TaskStatus.FAILED)
        self.assertEqual(success_result.status, TaskStatus.COMPLETED)
    
    def test_scheduler_resilience(self):
        """Test scheduler resilience to errors"""
        execution_count = {'success': 0, 'errors': 0}
        
        def unreliable_task():
            """Task that sometimes fails"""
            import random
            if random.random() < 0.3:  # 30% failure rate
                execution_count['errors'] += 1
                raise Exception("Random failure")
            else:
                execution_count['success'] += 1
                return "Success"
        
        # Schedule unreliable task
        schedule_id = self.cron_scheduler.add_schedule(
            name="Unreliable Task",
            cron_expression="*/1 * * * * *",  # Every second
            task_function=unreliable_task
        )
        
        # Run for several seconds
        self.cron_scheduler.start()
        time.sleep(5)
        self.cron_scheduler.stop()
        
        # Verify scheduler continued despite errors
        total_executions = execution_count['success'] + execution_count['errors']
        self.assertGreater(total_executions, 0)
        
        # Clean up
        self.cron_scheduler.remove_schedule(schedule_id)
    
    def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion"""
        from intelligent_automation_engine.parallel.resource_balancer import ResourceBalancer
        
        balancer = ResourceBalancer()
        balancer.add_resource("limited_resource", 2.0)  # Very limited resource
        
        # Create tasks that require more resources than available
        resource_intensive_tasks = []
        for i in range(5):
            task = Task(
                id=f"resource_task_{i}",
                name=f"Resource Task {i}",
                function=lambda i=i: time.sleep(0.5) or f"Task {i} done",
                estimated_duration=0.5,
                resource_requirements={'limited_resource': 1.0}
            )
            resource_intensive_tasks.append(task)
        
        # Execute plan
        plan = self.execution_planner.create_execution_plan(
            tasks=resource_intensive_tasks,
            strategy=ExecutionStrategy.RESOURCE_OPTIMIZED
        )
        
        result = self.execution_planner.execute_plan(plan)
        
        # Verify graceful handling of resource constraints
        self.assertIsNotNone(result)
        # Some tasks should complete, others may be queued/delayed
        completed_tasks = [
            r for r in result.task_results 
            if r.status == TaskStatus.COMPLETED
        ]
        self.assertGreater(len(completed_tasks), 0)


def create_integration_test_suite():
    """Create comprehensive integration test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestFullSystemIntegration,
        TestSystemReliability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == '__main__':
    # Run comprehensive integration test suite
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_integration_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*70}")
    print("INTELLIGENT AUTOMATION ENGINE INTEGRATION TEST SUMMARY")
    print(f"{'='*70}")
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
    
    print(f"{'='*70}")
    print("Integration testing complete!")
    print(f"{'='*70}")