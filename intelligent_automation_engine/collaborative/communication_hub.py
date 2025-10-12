"""
Communication Hub for Multi-Agent Collaboration

This module provides a centralized communication system for agents to exchange
messages, share data, coordinate activities, and maintain awareness of system state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any, Callable, Union
from datetime import datetime, timedelta
import uuid
import json
import asyncio
import logging
from abc import ABC, abstractmethod
import threading
from queue import Queue, PriorityQueue
import weakref

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the communication system"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    COORDINATION = "coordination"
    NOTIFICATION = "notification"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    SYSTEM = "system"
    USER = "user"


class MessagePriority(Enum):
    """Priority levels for messages"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class DeliveryMode(Enum):
    """Message delivery modes"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"
    FIRE_AND_FORGET = "fire_and_forget"


class MessageStatus(Enum):
    """Status of message delivery"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ChannelType(Enum):
    """Types of communication channels"""
    DIRECT = "direct"
    TOPIC = "topic"
    QUEUE = "queue"
    BROADCAST = "broadcast"
    SYSTEM = "system"
    EMERGENCY = "emergency"


@dataclass
class Message:
    """Represents a message in the communication system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Message identification
    type: MessageType = MessageType.INFO
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Routing information
    sender_id: str = ""
    recipient_id: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    channel: str = ""
    topic: str = ""
    
    # Message content
    subject: str = ""
    content: Any = None
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Delivery configuration
    delivery_mode: DeliveryMode = DeliveryMode.DIRECT
    require_acknowledgment: bool = False
    max_retries: int = 3
    retry_delay: timedelta = timedelta(seconds=1)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    # Status tracking
    status: MessageStatus = MessageStatus.PENDING
    delivery_attempts: int = 0
    error_message: str = ""
    
    # Message metadata
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Security and validation
    checksum: Optional[str] = None
    encrypted: bool = False
    signed: bool = False
    
    # Custom headers
    headers: Dict[str, str] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'recipients': self.recipients,
            'channel': self.channel,
            'topic': self.topic,
            'subject': self.subject,
            'content': self.content,
            'payload': self.payload,
            'delivery_mode': self.delivery_mode.value,
            'require_acknowledgment': self.require_acknowledgment,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'status': self.status.value,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'conversation_id': self.conversation_id,
            'headers': self.headers
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        message = cls()
        message.id = data.get('id', str(uuid.uuid4()))
        message.type = MessageType(data.get('type', 'info'))
        message.priority = MessagePriority(data.get('priority', 2))
        message.sender_id = data.get('sender_id', '')
        message.recipient_id = data.get('recipient_id')
        message.recipients = data.get('recipients', [])
        message.channel = data.get('channel', '')
        message.topic = data.get('topic', '')
        message.subject = data.get('subject', '')
        message.content = data.get('content')
        message.payload = data.get('payload', {})
        message.delivery_mode = DeliveryMode(data.get('delivery_mode', 'direct'))
        message.require_acknowledgment = data.get('require_acknowledgment', False)
        
        # Parse timestamps
        created_str = data.get('created_at')
        if created_str:
            message.created_at = datetime.fromisoformat(created_str)
        
        expires_str = data.get('expires_at')
        if expires_str:
            message.expires_at = datetime.fromisoformat(expires_str)
        
        message.status = MessageStatus(data.get('status', 'pending'))
        message.correlation_id = data.get('correlation_id')
        message.reply_to = data.get('reply_to')
        message.conversation_id = data.get('conversation_id')
        message.headers = data.get('headers', {})
        
        return message


@dataclass
class Channel:
    """Represents a communication channel"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: ChannelType = ChannelType.TOPIC
    
    # Channel configuration
    description: str = ""
    max_subscribers: int = 1000
    message_retention: timedelta = timedelta(hours=24)
    max_message_size: int = 1024 * 1024  # 1MB
    
    # Access control
    public: bool = True
    allowed_senders: Set[str] = field(default_factory=set)
    allowed_receivers: Set[str] = field(default_factory=set)
    moderators: Set[str] = field(default_factory=set)
    
    # Channel state
    active: bool = True
    subscribers: Set[str] = field(default_factory=set)
    message_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Channel metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subscription:
    """Represents a subscription to a channel or topic"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subscriber_id: str = ""
    channel_id: str = ""
    topic_pattern: str = ""
    
    # Subscription configuration
    message_filter: Optional[Callable[[Message], bool]] = None
    priority_filter: List[MessagePriority] = field(default_factory=list)
    type_filter: List[MessageType] = field(default_factory=list)
    
    # Delivery options
    delivery_mode: DeliveryMode = DeliveryMode.DIRECT
    batch_delivery: bool = False
    batch_size: int = 10
    batch_timeout: timedelta = timedelta(seconds=5)
    
    # Subscription state
    active: bool = True
    paused: bool = False
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    
    # Subscription metadata
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class CommunicationStats:
    """Statistics for communication hub"""
    total_messages_sent: int = 0
    total_messages_delivered: int = 0
    total_messages_failed: int = 0
    total_channels: int = 0
    total_subscriptions: int = 0
    active_connections: int = 0
    
    # Performance metrics
    average_delivery_time: timedelta = timedelta(0)
    message_throughput: float = 0.0  # messages per second
    peak_throughput: float = 0.0
    
    # Error tracking
    delivery_failures: int = 0
    timeout_errors: int = 0
    routing_errors: int = 0
    
    # Resource usage
    memory_usage: float = 0.0  # MB
    cpu_usage: float = 0.0  # percentage
    network_usage: float = 0.0  # MB/s


class MessageHandler(ABC):
    """Abstract base class for message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: Message) -> bool:
        """Handle an incoming message"""
        pass
    
    @abstractmethod
    def can_handle(self, message: Message) -> bool:
        """Check if this handler can process the message"""
        pass


class CommunicationHub:
    """Central communication hub for multi-agent collaboration"""
    
    def __init__(self):
        # Core components
        self.channels: Dict[str, Channel] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.message_handlers: List[MessageHandler] = []
        
        # Message queues
        self.message_queue = PriorityQueue()
        self.delivery_queue = Queue()
        self.acknowledgment_queue = Queue()
        
        # Message storage
        self.message_history: Dict[str, Message] = {}
        self.pending_messages: Dict[str, Message] = {}
        self.failed_messages: Dict[str, Message] = {}
        
        # Routing and delivery
        self.routing_table: Dict[str, str] = {}  # agent_id -> connection_id
        self.topic_subscribers: Dict[str, Set[str]] = {}
        self.message_filters: Dict[str, Callable[[Message], bool]] = {}
        
        # Connection management
        self.active_connections: Dict[str, Any] = {}
        self.connection_handlers: Dict[str, Callable] = {}
        
        # Background processing
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        self.delivery_thread: Optional[threading.Thread] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        
        # Configuration
        self.max_message_history = 10000
        self.message_retention_period = timedelta(hours=24)
        self.delivery_timeout = timedelta(seconds=30)
        self.max_retry_attempts = 3
        
        # Statistics
        self.stats = CommunicationStats()
        self.performance_metrics: List[Dict[str, Any]] = []
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'message_sent': [],
            'message_delivered': [],
            'message_failed': [],
            'channel_created': [],
            'subscription_added': [],
            'connection_established': [],
            'connection_lost': []
        }
        
        logger.info("Communication hub initialized")
    
    def start(self) -> bool:
        """Start the communication hub"""
        try:
            if self.running:
                return True
            
            self.running = True
            
            # Start worker threads
            self.delivery_thread = threading.Thread(target=self._delivery_worker, daemon=True)
            self.delivery_thread.start()
            
            self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            
            # Create default channels
            self._create_default_channels()
            
            logger.info("Communication hub started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start communication hub: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the communication hub"""
        try:
            self.running = False
            
            # Wait for threads to finish
            if self.delivery_thread and self.delivery_thread.is_alive():
                self.delivery_thread.join(timeout=5)
            
            if self.cleanup_thread and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=5)
            
            # Close connections
            for connection in self.active_connections.values():
                if hasattr(connection, 'close'):
                    connection.close()
            
            self.active_connections.clear()
            
            logger.info("Communication hub stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop communication hub: {e}")
            return False
    
    def send_message(self, message: Message) -> bool:
        """Send a message through the communication hub"""
        try:
            # Validate message
            if not self._validate_message(message):
                return False
            
            # Set message status
            message.status = MessageStatus.SENT
            message.delivery_attempts += 1
            
            # Store message
            self.message_history[message.id] = message
            self.pending_messages[message.id] = message
            
            # Route message based on delivery mode
            if message.delivery_mode == DeliveryMode.DIRECT:
                success = self._route_direct_message(message)
            elif message.delivery_mode == DeliveryMode.BROADCAST:
                success = self._route_broadcast_message(message)
            elif message.delivery_mode == DeliveryMode.MULTICAST:
                success = self._route_multicast_message(message)
            elif message.delivery_mode == DeliveryMode.PUBLISH_SUBSCRIBE:
                success = self._route_pubsub_message(message)
            else:
                success = self._route_direct_message(message)
            
            if success:
                self.stats.total_messages_sent += 1
                self._trigger_event('message_sent', message)
            else:
                message.status = MessageStatus.FAILED
                self.failed_messages[message.id] = message
                self.stats.total_messages_failed += 1
                self._trigger_event('message_failed', message)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def create_channel(self, name: str, channel_type: ChannelType = ChannelType.TOPIC, 
                      creator_id: str = "", **kwargs) -> Optional[Channel]:
        """Create a new communication channel"""
        try:
            channel = Channel(
                name=name,
                type=channel_type,
                created_by=creator_id,
                **kwargs
            )
            
            self.channels[channel.id] = channel
            self.stats.total_channels += 1
            
            # Initialize topic subscribers if needed
            if channel_type == ChannelType.TOPIC:
                self.topic_subscribers[channel.id] = set()
            
            self._trigger_event('channel_created', channel)
            
            logger.info(f"Channel created: {name} ({channel.id})")
            return channel
            
        except Exception as e:
            logger.error(f"Failed to create channel: {e}")
            return None
    
    def subscribe_to_channel(self, subscriber_id: str, channel_id: str, 
                           **kwargs) -> Optional[Subscription]:
        """Subscribe to a channel"""
        try:
            channel = self.channels.get(channel_id)
            if not channel:
                logger.error(f"Channel not found: {channel_id}")
                return None
            
            # Check permissions
            if not channel.public and subscriber_id not in channel.allowed_receivers:
                logger.error(f"Access denied to channel: {channel_id}")
                return None
            
            # Create subscription
            subscription = Subscription(
                subscriber_id=subscriber_id,
                channel_id=channel_id,
                **kwargs
            )
            
            self.subscriptions[subscription.id] = subscription
            channel.subscribers.add(subscriber_id)
            
            # Add to topic subscribers
            if channel.type == ChannelType.TOPIC:
                self.topic_subscribers[channel_id].add(subscriber_id)
            
            self.stats.total_subscriptions += 1
            self._trigger_event('subscription_added', subscription)
            
            logger.info(f"Subscription created: {subscriber_id} -> {channel_id}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to create subscription: {e}")
            return None
    
    def unsubscribe_from_channel(self, subscription_id: str) -> bool:
        """Unsubscribe from a channel"""
        try:
            subscription = self.subscriptions.get(subscription_id)
            if not subscription:
                return False
            
            # Remove from channel
            channel = self.channels.get(subscription.channel_id)
            if channel:
                channel.subscribers.discard(subscription.subscriber_id)
            
            # Remove from topic subscribers
            if subscription.channel_id in self.topic_subscribers:
                self.topic_subscribers[subscription.channel_id].discard(subscription.subscriber_id)
            
            # Remove subscription
            del self.subscriptions[subscription_id]
            self.stats.total_subscriptions -= 1
            
            logger.info(f"Subscription removed: {subscription_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove subscription: {e}")
            return False
    
    def register_agent(self, agent_id: str, connection_handler: Callable) -> bool:
        """Register an agent with the communication hub"""
        try:
            self.routing_table[agent_id] = agent_id
            self.connection_handlers[agent_id] = connection_handler
            self.stats.active_connections += 1
            
            self._trigger_event('connection_established', {'agent_id': agent_id})
            
            logger.info(f"Agent registered: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the communication hub"""
        try:
            # Remove from routing table
            self.routing_table.pop(agent_id, None)
            self.connection_handlers.pop(agent_id, None)
            
            # Remove from all subscriptions
            subscriptions_to_remove = []
            for sub_id, subscription in self.subscriptions.items():
                if subscription.subscriber_id == agent_id:
                    subscriptions_to_remove.append(sub_id)
            
            for sub_id in subscriptions_to_remove:
                self.unsubscribe_from_channel(sub_id)
            
            self.stats.active_connections -= 1
            self._trigger_event('connection_lost', {'agent_id': agent_id})
            
            logger.info(f"Agent unregistered: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent: {e}")
            return False
    
    def add_message_handler(self, handler: MessageHandler) -> bool:
        """Add a message handler"""
        try:
            self.message_handlers.append(handler)
            logger.info(f"Message handler added: {handler.__class__.__name__}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message handler: {e}")
            return False
    
    def remove_message_handler(self, handler: MessageHandler) -> bool:
        """Remove a message handler"""
        try:
            if handler in self.message_handlers:
                self.message_handlers.remove(handler)
                logger.info(f"Message handler removed: {handler.__class__.__name__}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove message handler: {e}")
            return False
    
    def get_channel_by_name(self, name: str) -> Optional[Channel]:
        """Get channel by name"""
        for channel in self.channels.values():
            if channel.name == name:
                return channel
        return None
    
    def get_agent_subscriptions(self, agent_id: str) -> List[Subscription]:
        """Get all subscriptions for an agent"""
        return [sub for sub in self.subscriptions.values() if sub.subscriber_id == agent_id]
    
    def get_channel_subscribers(self, channel_id: str) -> Set[str]:
        """Get all subscribers for a channel"""
        channel = self.channels.get(channel_id)
        return channel.subscribers if channel else set()
    
    def acknowledge_message(self, message_id: str, agent_id: str) -> bool:
        """Acknowledge receipt of a message"""
        try:
            message = self.pending_messages.get(message_id)
            if not message:
                return False
            
            message.status = MessageStatus.ACKNOWLEDGED
            message.acknowledged_at = datetime.now()
            
            # Remove from pending
            self.pending_messages.pop(message_id, None)
            
            logger.debug(f"Message acknowledged: {message_id} by {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge message: {e}")
            return False
    
    def _validate_message(self, message: Message) -> bool:
        """Validate a message before sending"""
        # Check required fields
        if not message.sender_id:
            logger.error("Message missing sender_id")
            return False
        
        # Check expiration
        if message.is_expired():
            logger.error("Message has expired")
            return False
        
        # Check recipients for direct messages
        if message.delivery_mode == DeliveryMode.DIRECT and not message.recipient_id:
            logger.error("Direct message missing recipient_id")
            return False
        
        return True
    
    def _route_direct_message(self, message: Message) -> bool:
        """Route a direct message to specific recipient"""
        try:
            recipient_id = message.recipient_id
            if not recipient_id:
                return False
            
            # Check if recipient is connected
            if recipient_id not in self.routing_table:
                logger.warning(f"Recipient not connected: {recipient_id}")
                return False
            
            # Deliver message
            return self._deliver_message_to_agent(message, recipient_id)
            
        except Exception as e:
            logger.error(f"Failed to route direct message: {e}")
            return False
    
    def _route_broadcast_message(self, message: Message) -> bool:
        """Route a broadcast message to all connected agents"""
        try:
            success_count = 0
            
            for agent_id in self.routing_table.keys():
                if agent_id != message.sender_id:  # Don't send to sender
                    if self._deliver_message_to_agent(message, agent_id):
                        success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to route broadcast message: {e}")
            return False
    
    def _route_multicast_message(self, message: Message) -> bool:
        """Route a multicast message to specific recipients"""
        try:
            success_count = 0
            
            for recipient_id in message.recipients:
                if recipient_id in self.routing_table:
                    if self._deliver_message_to_agent(message, recipient_id):
                        success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to route multicast message: {e}")
            return False
    
    def _route_pubsub_message(self, message: Message) -> bool:
        """Route a publish-subscribe message to topic subscribers"""
        try:
            success_count = 0
            
            # Find subscribers for the topic/channel
            subscribers = set()
            
            if message.channel:
                channel = self.get_channel_by_name(message.channel)
                if channel:
                    subscribers.update(channel.subscribers)
            
            if message.topic and message.topic in self.topic_subscribers:
                subscribers.update(self.topic_subscribers[message.topic])
            
            # Deliver to subscribers
            for subscriber_id in subscribers:
                if subscriber_id != message.sender_id:  # Don't send to sender
                    if self._deliver_message_to_agent(message, subscriber_id):
                        success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to route pub-sub message: {e}")
            return False
    
    def _deliver_message_to_agent(self, message: Message, agent_id: str) -> bool:
        """Deliver a message to a specific agent"""
        try:
            # Get connection handler
            handler = self.connection_handlers.get(agent_id)
            if not handler:
                return False
            
            # Apply message filters
            if not self._apply_message_filters(message, agent_id):
                return False
            
            # Process through message handlers
            for msg_handler in self.message_handlers:
                if msg_handler.can_handle(message):
                    asyncio.create_task(msg_handler.handle_message(message))
            
            # Deliver message
            success = handler(message)
            
            if success:
                message.status = MessageStatus.DELIVERED
                message.delivered_at = datetime.now()
                self.stats.total_messages_delivered += 1
                self._trigger_event('message_delivered', message)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to deliver message to {agent_id}: {e}")
            return False
    
    def _apply_message_filters(self, message: Message, agent_id: str) -> bool:
        """Apply message filters for an agent"""
        try:
            # Find relevant subscriptions
            for subscription in self.subscriptions.values():
                if subscription.subscriber_id == agent_id and subscription.active:
                    # Apply priority filter
                    if subscription.priority_filter and message.priority not in subscription.priority_filter:
                        continue
                    
                    # Apply type filter
                    if subscription.type_filter and message.type not in subscription.type_filter:
                        continue
                    
                    # Apply custom filter
                    if subscription.message_filter and not subscription.message_filter(message):
                        continue
                    
                    return True
            
            # If no subscription found, allow by default
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply message filters: {e}")
            return True
    
    def _create_default_channels(self) -> None:
        """Create default communication channels"""
        default_channels = [
            ("system", ChannelType.SYSTEM),
            ("general", ChannelType.TOPIC),
            ("coordination", ChannelType.TOPIC),
            ("status", ChannelType.TOPIC),
            ("emergency", ChannelType.EMERGENCY)
        ]
        
        for name, channel_type in default_channels:
            self.create_channel(name, channel_type, "system")
    
    def _delivery_worker(self) -> None:
        """Background worker for message delivery"""
        while self.running:
            try:
                # Process delivery queue
                if not self.delivery_queue.empty():
                    message = self.delivery_queue.get(timeout=1)
                    # Process message delivery
                    self._process_message_delivery(message)
                
                # Process acknowledgment queue
                if not self.acknowledgment_queue.empty():
                    ack_data = self.acknowledgment_queue.get(timeout=1)
                    # Process acknowledgment
                    self._process_acknowledgment(ack_data)
                
            except Exception as e:
                if self.running:  # Only log if we're still supposed to be running
                    logger.error(f"Delivery worker error: {e}")
            
            # Small delay to prevent busy waiting
            threading.Event().wait(0.1)
    
    def _cleanup_worker(self) -> None:
        """Background worker for cleanup tasks"""
        while self.running:
            try:
                # Clean up expired messages
                self._cleanup_expired_messages()
                
                # Clean up old message history
                self._cleanup_message_history()
                
                # Update performance metrics
                self._update_performance_metrics()
                
            except Exception as e:
                if self.running:
                    logger.error(f"Cleanup worker error: {e}")
            
            # Run cleanup every minute
            threading.Event().wait(60)
    
    def _cleanup_expired_messages(self) -> None:
        """Clean up expired messages"""
        current_time = datetime.now()
        expired_messages = []
        
        for message_id, message in self.pending_messages.items():
            if message.is_expired():
                expired_messages.append(message_id)
        
        for message_id in expired_messages:
            message = self.pending_messages.pop(message_id, None)
            if message:
                message.status = MessageStatus.EXPIRED
                self.failed_messages[message_id] = message
    
    def _cleanup_message_history(self) -> None:
        """Clean up old message history"""
        if len(self.message_history) <= self.max_message_history:
            return
        
        # Sort by creation time and keep only recent messages
        sorted_messages = sorted(
            self.message_history.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )
        
        # Keep only the most recent messages
        self.message_history = dict(sorted_messages[:self.max_message_history])
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        current_time = datetime.now()
        
        # Calculate message throughput
        recent_messages = [
            msg for msg in self.message_history.values()
            if current_time - msg.created_at <= timedelta(minutes=1)
        ]
        
        self.stats.message_throughput = len(recent_messages) / 60.0  # messages per second
        self.stats.peak_throughput = max(self.stats.peak_throughput, self.stats.message_throughput)
        
        # Store metrics
        metrics = {
            'timestamp': current_time,
            'throughput': self.stats.message_throughput,
            'active_connections': self.stats.active_connections,
            'pending_messages': len(self.pending_messages),
            'failed_messages': len(self.failed_messages)
        }
        
        self.performance_metrics.append(metrics)
        
        # Keep only recent metrics
        cutoff_time = current_time - timedelta(hours=1)
        self.performance_metrics = [
            m for m in self.performance_metrics
            if m['timestamp'] > cutoff_time
        ]
    
    def _process_message_delivery(self, message: Message) -> None:
        """Process message delivery"""
        # Implementation would handle actual message delivery
        pass
    
    def _process_acknowledgment(self, ack_data: Dict[str, Any]) -> None:
        """Process message acknowledgment"""
        # Implementation would handle acknowledgment processing
        pass
    
    def _trigger_event(self, event_name: str, data: Any) -> None:
        """Trigger event callbacks"""
        callbacks = self.event_callbacks.get(event_name, [])
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def add_event_callback(self, event_name: str, callback: Callable) -> bool:
        """Add event callback"""
        if event_name in self.event_callbacks:
            self.event_callbacks[event_name].append(callback)
            return True
        return False
    
    def remove_event_callback(self, event_name: str, callback: Callable) -> bool:
        """Remove event callback"""
        if event_name in self.event_callbacks and callback in self.event_callbacks[event_name]:
            self.event_callbacks[event_name].remove(callback)
            return True
        return False
    
    def get_statistics(self) -> CommunicationStats:
        """Get communication statistics"""
        return self.stats
    
    def get_performance_metrics(self) -> List[Dict[str, Any]]:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'running': self.running,
            'channels': len(self.channels),
            'subscriptions': len(self.subscriptions),
            'active_connections': len(self.active_connections),
            'pending_messages': len(self.pending_messages),
            'failed_messages': len(self.failed_messages),
            'message_history_size': len(self.message_history),
            'stats': self.stats.__dict__
        }