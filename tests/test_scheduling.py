"""
Comprehensive Integration Tests for Scheduling Components

This module provides comprehensive integration tests for all scheduling
components including CronScheduler, NaturalLanguageScheduler, ScheduleOptimizer,
TimeManager, and EventScheduler.
"""

import unittest
import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import tempfile
import os
import json

# Import components to test
from intelligent_automation_engine.scheduling import (
    CronScheduler,
    NaturalLanguageScheduler,
    ScheduleOptimizer,
    TimeManager,
    EventScheduler,
    # Data structures and enums
    CronExpression,
    ScheduleEntry,
    ScheduleStatus,
    TimeExpression,
    ParsedSchedule,
    ScheduleConflict,
    OptimizationResult,
    TimeRange,
    BusinessHours,
    Event,
    EventTrigger,
    EventHandler,
    EventType,
    EventPriority,
    EventStatus,
    TriggerType,
    SchedulerState
)


class TestCronScheduler(unittest.TestCase):
    """Test cases for CronScheduler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = CronScheduler()
        self.test_results = []
    
    def tearDown(self):
        """Clean up test fixtures"""
        if self.scheduler.is_running():
            self.scheduler.stop()
    
    def test_cron_expression_parsing(self):
        """Test cron expression parsing"""
        test_expressions = [
            "0 9 * * 1-5",  # 9 AM on weekdays
            "*/15 * * * *",  # Every 15 minutes
            "0 0 1 * *",     # First day of every month
            "0 12 * * 0",    # Noon on Sundays
            "30 14 * * 2",   # 2:30 PM on Tuesdays
        ]
        
        for expr_str in test_expressions:
            with self.subTest(expression=expr_str):
                expr = self.scheduler.parse_cron_expression(expr_str)
                self.assertIsInstance(expr, CronExpression)
                self.assertEqual(expr.expression, expr_str)
    
    def test_schedule_management(self):
        """Test schedule management operations"""
        # Add a schedule
        def test_task():
            self.test_results.append(f"Task executed at {datetime.now()}")
        
        schedule_id = self.scheduler.add_schedule(
            name="test_schedule",
            cron_expression="*/1 * * * *",  # Every minute
            task_function=test_task,
            description="Test schedule"
        )
        
        self.assertIsNotNone(schedule_id)
        
        # Check schedule status
        status = self.scheduler.get_schedule_status(schedule_id)
        self.assertEqual(status, ScheduleStatus.ENABLED)
        
        # Disable schedule
        success = self.scheduler.disable_schedule(schedule_id)
        self.assertTrue(success)
        
        status = self.scheduler.get_schedule_status(schedule_id)
        self.assertEqual(status, ScheduleStatus.DISABLED)
        
        # Remove schedule
        success = self.scheduler.remove_schedule(schedule_id)
        self.assertTrue(success)
    
    def test_scheduler_execution(self):
        """Test scheduler execution"""
        execution_count = {'count': 0}
        
        def test_task():
            execution_count['count'] += 1
        
        # Add a schedule that runs every 2 seconds
        schedule_id = self.scheduler.add_schedule(
            name="execution_test",
            cron_expression="*/2 * * * * *",  # Every 2 seconds
            task_function=test_task
        )
        
        # Start scheduler
        self.scheduler.start()
        
        # Wait for a few executions
        time.sleep(5)
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Verify executions occurred
        self.assertGreater(execution_count['count'], 0)
        
        # Clean up
        self.scheduler.remove_schedule(schedule_id)
    
    def test_next_run_calculation(self):
        """Test next run time calculation"""
        # Test daily at 9 AM
        expr = self.scheduler.parse_cron_expression("0 9 * * *")
        next_run = self.scheduler.get_next_run_time("test", expr)
        
        self.assertIsInstance(next_run, datetime)
        self.assertEqual(next_run.hour, 9)
        self.assertEqual(next_run.minute, 0)


class TestNaturalLanguageScheduler(unittest.TestCase):
    """Test cases for NaturalLanguageScheduler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = NaturalLanguageScheduler()
    
    def test_natural_language_parsing(self):
        """Test natural language expression parsing"""
        test_expressions = [
            "every day at 9 AM",
            "every Monday at 2 PM",
            "every 30 minutes",
            "every hour",
            "daily at noon",
            "weekly on Friday at 5 PM",
            "monthly on the 1st at 10 AM",
            "in 2 hours",
            "tomorrow at 3 PM",
            "next week on Tuesday"
        ]
        
        for expr in test_expressions:
            with self.subTest(expression=expr):
                parsed = self.scheduler.parse_expression(expr)
                self.assertIsInstance(parsed, ParsedSchedule)
                self.assertIsNotNone(parsed.next_execution)
    
    def test_schedule_suggestions(self):
        """Test schedule suggestions"""
        suggestions = self.scheduler.get_schedule_suggestions("meeting")
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)
    
    def test_expression_validation(self):
        """Test expression validation"""
        valid_expressions = [
            "every day at 9 AM",
            "every Monday",
            "hourly"
        ]
        
        invalid_expressions = [
            "invalid expression",
            "every 25 hours",  # Invalid interval
            "on the 32nd"      # Invalid day
        ]
        
        for expr in valid_expressions:
            with self.subTest(expression=expr, valid=True):
                is_valid = self.scheduler.validate_expression(expr)
                self.assertTrue(is_valid)
        
        for expr in invalid_expressions:
            with self.subTest(expression=expr, valid=False):
                is_valid = self.scheduler.validate_expression(expr)
                self.assertFalse(is_valid)


class TestScheduleOptimizer(unittest.TestCase):
    """Test cases for ScheduleOptimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = ScheduleOptimizer()
    
    def test_conflict_detection(self):
        """Test schedule conflict detection"""
        # Add overlapping schedules
        schedule1_id = self.optimizer.add_schedule(
            name="Meeting 1",
            start_time=datetime.now() + timedelta(hours=1),
            duration=timedelta(hours=1),
            resource_requirements={'room': 'Conference Room A'}
        )
        
        schedule2_id = self.optimizer.add_schedule(
            name="Meeting 2",
            start_time=datetime.now() + timedelta(hours=1, minutes=30),
            duration=timedelta(hours=1),
            resource_requirements={'room': 'Conference Room A'}
        )
        
        # Detect conflicts
        conflicts = self.optimizer.detect_conflicts()
        
        self.assertGreater(len(conflicts), 0)
        
        # Verify conflict details
        conflict = conflicts[0]
        self.assertIsInstance(conflict, ScheduleConflict)
        self.assertIn(schedule1_id, [conflict.schedule1_id, conflict.schedule2_id])
        self.assertIn(schedule2_id, [conflict.schedule1_id, conflict.schedule2_id])
    
    def test_schedule_optimization(self):
        """Test schedule optimization"""
        # Add multiple schedules
        schedule_ids = []
        base_time = datetime.now() + timedelta(hours=1)
        
        for i in range(5):
            schedule_id = self.optimizer.add_schedule(
                name=f"Task {i}",
                start_time=base_time + timedelta(minutes=i*30),
                duration=timedelta(minutes=45),
                priority=i % 3,
                resource_requirements={'cpu': 0.5}
            )
            schedule_ids.append(schedule_id)
        
        # Optimize schedules
        result = self.optimizer.optimize_schedules()
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)
        self.assertGreater(result.improvement_score, 0)
    
    def test_conflict_resolution(self):
        """Test conflict resolution"""
        # Create conflicting schedules
        schedule1_id = self.optimizer.add_schedule(
            name="High Priority Task",
            start_time=datetime.now() + timedelta(hours=2),
            duration=timedelta(hours=1),
            priority=3,
            resource_requirements={'worker': 'Worker1'}
        )
        
        schedule2_id = self.optimizer.add_schedule(
            name="Low Priority Task",
            start_time=datetime.now() + timedelta(hours=2, minutes=30),
            duration=timedelta(hours=1),
            priority=1,
            resource_requirements={'worker': 'Worker1'}
        )
        
        # Detect and resolve conflicts
        conflicts = self.optimizer.detect_conflicts()
        self.assertGreater(len(conflicts), 0)
        
        resolution = self.optimizer.resolve_conflict(conflicts[0])
        self.assertIsNotNone(resolution)


class TestTimeManager(unittest.TestCase):
    """Test cases for TimeManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.time_manager = TimeManager()
    
    def test_timezone_operations(self):
        """Test timezone operations"""
        # Test timezone conversion
        utc_time = datetime.now(timezone.utc)
        est_time = self.time_manager.convert_timezone(utc_time, 'US/Eastern')
        
        self.assertIsInstance(est_time, datetime)
        self.assertNotEqual(utc_time.hour, est_time.hour)
    
    def test_business_hours(self):
        """Test business hours functionality"""
        # Set business hours
        business_hours = BusinessHours(
            start_time=datetime.strptime("09:00", "%H:%M").time(),
            end_time=datetime.strptime("17:00", "%H:%M").time(),
            days_of_week=[0, 1, 2, 3, 4]  # Monday to Friday
        )
        
        self.time_manager.add_business_hours("default", business_hours)
        
        # Test business hours checking
        # Create a Monday at 10 AM
        monday_10am = datetime(2024, 1, 8, 10, 0)  # Assuming this is a Monday
        is_business_time = self.time_manager.is_business_time(monday_10am, "default")
        
        # Create a Saturday at 10 AM
        saturday_10am = datetime(2024, 1, 13, 10, 0)  # Assuming this is a Saturday
        is_weekend = self.time_manager.is_business_time(saturday_10am, "default")
        
        self.assertTrue(is_business_time)
        self.assertFalse(is_weekend)
    
    def test_time_calculations(self):
        """Test time calculations"""
        base_time = datetime.now()
        
        # Test duration calculation
        end_time = base_time + timedelta(hours=2, minutes=30)
        duration = self.time_manager.calculate_duration(base_time, end_time)
        
        self.assertEqual(duration, timedelta(hours=2, minutes=30))
        
        # Test time addition
        future_time = self.time_manager.add_time(base_time, hours=3, minutes=15)
        expected_time = base_time + timedelta(hours=3, minutes=15)
        
        self.assertEqual(future_time, expected_time)
    
    def test_time_ranges(self):
        """Test time range operations"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=2)
        
        range1 = self.time_manager.create_time_range(start_time, end_time)
        
        # Test overlapping range
        overlap_start = start_time + timedelta(hours=1)
        overlap_end = end_time + timedelta(hours=1)
        range2 = self.time_manager.create_time_range(overlap_start, overlap_end)
        
        # Test overlap detection
        has_overlap = self.time_manager.check_time_overlap(range1, range2)
        self.assertTrue(has_overlap)
        
        # Test intersection
        intersection = self.time_manager.get_time_intersection(range1, range2)
        self.assertIsNotNone(intersection)


class TestEventScheduler(unittest.TestCase):
    """Test cases for EventScheduler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scheduler = EventScheduler()
        self.event_results = []
    
    def tearDown(self):
        """Clean up test fixtures"""
        if self.scheduler.state == SchedulerState.RUNNING:
            self.scheduler.stop()
    
    def test_event_scheduling(self):
        """Test basic event scheduling"""
        # Create test event
        event = Event(
            name="Test Event",
            event_type=EventType.USER,
            data={'message': 'Hello World'},
            priority=EventPriority.HIGH
        )
        
        # Start scheduler
        self.scheduler.start()
        
        # Schedule event
        success = self.scheduler.schedule_event(event)
        self.assertTrue(success)
        
        # Wait for processing
        time.sleep(1)
        
        # Check event status
        status = self.scheduler.get_event_status(event.id)
        self.assertIn(status, [EventStatus.COMPLETED, EventStatus.PROCESSING])
        
        # Stop scheduler
        self.scheduler.stop()
    
    def test_event_triggers(self):
        """Test event triggers"""
        trigger_executed = {'count': 0}
        
        def trigger_action(event):
            trigger_executed['count'] += 1
            return f"Triggered: {event.name}"
        
        # Create event trigger
        trigger = EventTrigger(
            name="Test Trigger",
            trigger_type=TriggerType.TIME_BASED,
            schedule="2s",  # Every 2 seconds
            actions=[{'function': trigger_action}],
            event_template={'name': 'Triggered Event', 'type': 'triggered'}
        )
        
        # Add trigger
        success = self.scheduler.add_trigger(trigger)
        self.assertTrue(success)
        
        # Start scheduler
        self.scheduler.start()
        
        # Wait for trigger executions
        time.sleep(5)
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Verify trigger executed
        self.assertGreater(trigger_executed['count'], 0)
    
    def test_event_handlers(self):
        """Test event handlers"""
        handler_executed = {'count': 0}
        
        def event_handler(event):
            handler_executed['count'] += 1
            return f"Handled: {event.name}"
        
        # Create event handler
        handler = EventHandler(
            name="Test Handler",
            event_types=[EventType.USER],
            handler_function=event_handler
        )
        
        # Add handler
        success = self.scheduler.add_handler(handler)
        self.assertTrue(success)
        
        # Start scheduler
        self.scheduler.start()
        
        # Create and schedule event
        event = Event(
            name="Test Event",
            event_type=EventType.USER,
            data={'test': True}
        )
        
        self.scheduler.schedule_event(event)
        
        # Wait for processing
        time.sleep(1)
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Verify handler executed
        self.assertGreater(handler_executed['count'], 0)
    
    def test_event_subscriptions(self):
        """Test event subscriptions"""
        from intelligent_automation_engine.scheduling.event_scheduler import EventSubscription
        
        # Create subscription
        subscription = EventSubscription(
            subscriber_id="test_subscriber",
            subscriber_name="Test Subscriber",
            event_types=[EventType.SYSTEM],
            delivery_method="callback"
        )
        
        # Add subscription
        success = self.scheduler.subscribe(subscription)
        self.assertTrue(success)
        
        # Start scheduler
        self.scheduler.start()
        
        # Create system event
        event = Event(
            name="System Event",
            event_type=EventType.SYSTEM,
            data={'system_info': 'test'}
        )
        
        self.scheduler.schedule_event(event)
        
        # Wait for processing
        time.sleep(1)
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Verify subscription is active
        self.assertTrue(subscription.is_active)


class TestIntegratedScheduling(unittest.TestCase):
    """Integration tests for all scheduling components working together"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.cron_scheduler = CronScheduler()
        self.nl_scheduler = NaturalLanguageScheduler()
        self.optimizer = ScheduleOptimizer()
        self.time_manager = TimeManager()
        self.event_scheduler = EventScheduler()
    
    def tearDown(self):
        """Clean up test environment"""
        if self.cron_scheduler.is_running():
            self.cron_scheduler.stop()
        if self.event_scheduler.state == SchedulerState.RUNNING:
            self.event_scheduler.stop()
    
    def test_end_to_end_scheduling_workflow(self):
        """Test end-to-end scheduling workflow"""
        execution_log = []
        
        def log_execution(task_name):
            execution_log.append({
                'task': task_name,
                'time': datetime.now(),
                'thread': threading.current_thread().name
            })
        
        # 1. Parse natural language schedule
        nl_expression = "every 3 seconds"
        parsed_schedule = self.nl_scheduler.parse_expression(nl_expression)
        self.assertIsNotNone(parsed_schedule)
        
        # 2. Convert to cron expression (simplified)
        cron_expr = "*/3 * * * * *"  # Every 3 seconds
        
        # 3. Add to cron scheduler
        schedule_id = self.cron_scheduler.add_schedule(
            name="NL Parsed Task",
            cron_expression=cron_expr,
            task_function=lambda: log_execution("NL Parsed Task")
        )
        
        # 4. Add to optimizer for conflict checking
        optimizer_schedule_id = self.optimizer.add_schedule(
            name="NL Parsed Task",
            start_time=datetime.now(),
            duration=timedelta(seconds=1),
            resource_requirements={'cpu': 0.1}
        )
        
        # 5. Check for conflicts
        conflicts = self.optimizer.detect_conflicts()
        # Should be no conflicts with single schedule
        
        # 6. Start execution
        self.cron_scheduler.start()
        
        # 7. Wait for executions
        time.sleep(8)
        
        # 8. Stop scheduler
        self.cron_scheduler.stop()
        
        # 9. Verify execution
        self.assertGreater(len(execution_log), 1)
        
        # 10. Clean up
        self.cron_scheduler.remove_schedule(schedule_id)
    
    def test_complex_scheduling_scenario(self):
        """Test complex scheduling scenario with multiple components"""
        results = {'events': [], 'cron_tasks': [], 'conflicts': []}
        
        # 1. Set up business hours
        business_hours = BusinessHours(
            start_time=datetime.strptime("09:00", "%H:%M").time(),
            end_time=datetime.strptime("17:00", "%H:%M").time(),
            days_of_week=[0, 1, 2, 3, 4]
        )
        self.time_manager.add_business_hours("office", business_hours)
        
        # 2. Create multiple schedules with potential conflicts
        base_time = datetime.now() + timedelta(minutes=1)
        
        # Schedule 1: High priority meeting
        meeting_id = self.optimizer.add_schedule(
            name="Important Meeting",
            start_time=base_time,
            duration=timedelta(minutes=30),
            priority=3,
            resource_requirements={'room': 'Conference Room', 'projector': True}
        )
        
        # Schedule 2: Overlapping low priority task
        task_id = self.optimizer.add_schedule(
            name="Regular Task",
            start_time=base_time + timedelta(minutes=15),
            duration=timedelta(minutes=30),
            priority=1,
            resource_requirements={'room': 'Conference Room'}
        )
        
        # 3. Detect and resolve conflicts
        conflicts = self.optimizer.detect_conflicts()
        results['conflicts'] = conflicts
        
        if conflicts:
            for conflict in conflicts:
                resolution = self.optimizer.resolve_conflict(conflict)
                self.assertIsNotNone(resolution)
        
        # 4. Set up event-driven scheduling
        def event_handler(event):
            results['events'].append(event.name)
        
        handler = EventHandler(
            name="Schedule Handler",
            event_types=[EventType.SCHEDULED],
            handler_function=event_handler
        )
        
        self.event_scheduler.add_handler(handler)
        self.event_scheduler.start()
        
        # 5. Create scheduled events
        for i in range(3):
            event = Event(
                name=f"Scheduled Event {i}",
                event_type=EventType.SCHEDULED,
                scheduled_at=datetime.now() + timedelta(seconds=i+1)
            )
            self.event_scheduler.schedule_event(event)
        
        # 6. Wait for processing
        time.sleep(5)
        
        # 7. Verify results
        self.assertGreater(len(results['events']), 0)
        self.assertGreater(len(results['conflicts']), 0)
        
        # 8. Clean up
        self.event_scheduler.stop()
    
    def test_timezone_aware_scheduling(self):
        """Test timezone-aware scheduling"""
        # Create schedules in different timezones
        utc_time = datetime.now(timezone.utc)
        
        # Convert to different timezones
        est_time = self.time_manager.convert_timezone(utc_time, 'US/Eastern')
        pst_time = self.time_manager.convert_timezone(utc_time, 'US/Pacific')
        
        # Verify timezone conversions
        self.assertNotEqual(utc_time.hour, est_time.hour)
        self.assertNotEqual(est_time.hour, pst_time.hour)
        
        # Schedule tasks in different timezones
        execution_times = []
        
        def record_execution(tz_name):
            execution_times.append({
                'timezone': tz_name,
                'time': datetime.now(),
                'utc_time': datetime.now(timezone.utc)
            })
        
        # Add timezone-specific schedules
        est_schedule = self.cron_scheduler.add_schedule(
            name="EST Task",
            cron_expression="*/2 * * * * *",
            task_function=lambda: record_execution("EST"),
            timezone_str='US/Eastern'
        )
        
        pst_schedule = self.cron_scheduler.add_schedule(
            name="PST Task", 
            cron_expression="*/2 * * * * *",
            task_function=lambda: record_execution("PST"),
            timezone_str='US/Pacific'
        )
        
        # Start scheduler
        self.cron_scheduler.start()
        
        # Wait for executions
        time.sleep(5)
        
        # Stop scheduler
        self.cron_scheduler.stop()
        
        # Verify executions
        self.assertGreater(len(execution_times), 0)
        
        # Clean up
        self.cron_scheduler.remove_schedule(est_schedule)
        self.cron_scheduler.remove_schedule(pst_schedule)


class TestSchedulingPerformance(unittest.TestCase):
    """Performance tests for scheduling components"""
    
    def test_large_schedule_set_performance(self):
        """Test performance with large number of schedules"""
        optimizer = ScheduleOptimizer()
        
        # Add large number of schedules
        num_schedules = 1000
        schedule_ids = []
        
        start_time = time.time()
        base_time = datetime.now() + timedelta(hours=1)
        
        for i in range(num_schedules):
            schedule_id = optimizer.add_schedule(
                name=f"Schedule {i}",
                start_time=base_time + timedelta(minutes=i),
                duration=timedelta(minutes=30),
                priority=i % 5
            )
            schedule_ids.append(schedule_id)
        
        creation_time = time.time() - start_time
        
        # Test conflict detection performance
        start_time = time.time()
        conflicts = optimizer.detect_conflicts()
        detection_time = time.time() - start_time
        
        # Verify reasonable performance
        self.assertLess(creation_time, 10.0)  # Should create 1000 schedules in under 10 seconds
        self.assertLess(detection_time, 30.0)  # Should detect conflicts in under 30 seconds
        
        print(f"Created {num_schedules} schedules in {creation_time:.2f}s")
        print(f"Detected {len(conflicts)} conflicts in {detection_time:.2f}s")
    
    def test_concurrent_scheduling_performance(self):
        """Test concurrent scheduling performance"""
        event_scheduler = EventScheduler(max_workers=5)
        event_scheduler.start()
        
        # Schedule many events concurrently
        num_events = 100
        events = []
        
        start_time = time.time()
        
        for i in range(num_events):
            event = Event(
                name=f"Concurrent Event {i}",
                event_type=EventType.USER,
                data={'index': i},
                priority=EventPriority.MEDIUM
            )
            events.append(event)
            event_scheduler.schedule_event(event)
        
        # Wait for all events to process
        time.sleep(5)
        
        processing_time = time.time() - start_time
        
        # Get metrics
        metrics = event_scheduler.get_scheduler_metrics()
        
        # Stop scheduler
        event_scheduler.stop()
        
        # Verify performance
        self.assertLess(processing_time, 10.0)  # Should process 100 events in under 10 seconds
        self.assertGreater(metrics.processed_events, 0)
        
        print(f"Processed {metrics.processed_events} events in {processing_time:.2f}s")


def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCronScheduler,
        TestNaturalLanguageScheduler,
        TestScheduleOptimizer,
        TestTimeManager,
        TestEventScheduler,
        TestIntegratedScheduling,
        TestSchedulingPerformance
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
    print("SCHEDULING INTEGRATION TEST SUMMARY")
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