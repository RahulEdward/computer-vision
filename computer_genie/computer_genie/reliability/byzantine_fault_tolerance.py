"""
Byzantine Fault Tolerance for Distributed Consensus

Implements Byzantine fault-tolerant consensus algorithms to ensure system
reliability even when some nodes behave maliciously or fail arbitrarily.
"""

import asyncio
import time
import hashlib
import json
import logging
import threading
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict, Counter
import hmac
import secrets

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Node states in the consensus protocol"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    FAULTY = "faulty"
    OFFLINE = "offline"


class MessageType(Enum):
    """Types of consensus messages"""
    PREPARE = "prepare"
    PROMISE = "promise"
    ACCEPT = "accept"
    ACCEPTED = "accepted"
    COMMIT = "commit"
    HEARTBEAT = "heartbeat"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_RESPONSE = "append_response"


class ConsensusPhase(Enum):
    """Phases of the consensus protocol"""
    PREPARE_PHASE = "prepare"
    ACCEPT_PHASE = "accept"
    COMMIT_PHASE = "commit"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ConsensusMessage:
    """Message in the consensus protocol"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.PREPARE
    sender_id: str = ""
    receiver_id: Optional[str] = None
    term: int = 0
    proposal_id: str = ""
    value: Any = None
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """Consensus proposal"""
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposer_id: str = ""
    value: Any = None
    term: int = 0
    timestamp: float = field(default_factory=time.time)
    phase: ConsensusPhase = ConsensusPhase.PREPARE_PHASE
    promises: Set[str] = field(default_factory=set)
    accepts: Set[str] = field(default_factory=set)
    commits: Set[str] = field(default_factory=set)
    required_majority: int = 0


@dataclass
class NodeInfo:
    """Information about a consensus node"""
    node_id: str
    address: str
    port: int
    state: NodeState = NodeState.FOLLOWER
    last_heartbeat: float = field(default_factory=time.time)
    term: int = 0
    voted_for: Optional[str] = None
    is_trusted: bool = True
    failure_count: int = 0
    public_key: Optional[str] = None


class ByzantineDetector:
    """Detects Byzantine (malicious or faulty) behavior"""
    
    def __init__(self, detection_threshold: float = 0.7):
        self.detection_threshold = detection_threshold
        self.node_behaviors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.suspicious_patterns: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        
    def record_behavior(self, node_id: str, behavior: Dict[str, Any]):
        """Record node behavior for analysis"""
        with self._lock:
            behavior['timestamp'] = time.time()
            self.node_behaviors[node_id].append(behavior)
            
            # Keep only recent behaviors (last 1000)
            if len(self.node_behaviors[node_id]) > 1000:
                self.node_behaviors[node_id] = self.node_behaviors[node_id][-1000:]
                
    def detect_byzantine_behavior(self, node_id: str) -> bool:
        """Detect if a node is exhibiting Byzantine behavior"""
        with self._lock:
            behaviors = self.node_behaviors.get(node_id, [])
            if len(behaviors) < 10:  # Need sufficient data
                return False
                
            # Analyze recent behaviors
            recent_behaviors = [b for b in behaviors if time.time() - b['timestamp'] < 300]  # Last 5 minutes
            
            if len(recent_behaviors) < 5:
                return False
                
            # Check for suspicious patterns
            suspicion_score = 0.0
            
            # Pattern 1: Inconsistent voting
            votes = [b.get('vote') for b in recent_behaviors if 'vote' in b]
            if len(votes) > 2:
                vote_changes = sum(1 for i in range(1, len(votes)) if votes[i] != votes[i-1])
                if vote_changes / len(votes) > 0.5:  # Too many vote changes
                    suspicion_score += 0.3
                    
            # Pattern 2: Message timing anomalies
            message_times = [b.get('response_time', 0) for b in recent_behaviors if 'response_time' in b]
            if message_times:
                avg_time = sum(message_times) / len(message_times)
                outliers = sum(1 for t in message_times if abs(t - avg_time) > avg_time * 2)
                if outliers / len(message_times) > 0.3:  # Too many timing outliers
                    suspicion_score += 0.2
                    
            # Pattern 3: Conflicting messages
            conflicting_messages = sum(1 for b in recent_behaviors if b.get('conflicting', False))
            if conflicting_messages / len(recent_behaviors) > 0.2:  # Too many conflicts
                suspicion_score += 0.4
                
            # Pattern 4: Signature verification failures
            signature_failures = sum(1 for b in recent_behaviors if b.get('signature_valid', True) is False)
            if signature_failures > 0:
                suspicion_score += 0.5
                
            # Update suspicious patterns count
            if suspicion_score > self.detection_threshold:
                self.suspicious_patterns[node_id] += 1
                
            return suspicion_score > self.detection_threshold
            
    def get_node_trust_score(self, node_id: str) -> float:
        """Get trust score for a node (0.0 = untrusted, 1.0 = fully trusted)"""
        with self._lock:
            behaviors = self.node_behaviors.get(node_id, [])
            if not behaviors:
                return 1.0  # Default trust for new nodes
                
            # Calculate trust based on recent behavior
            recent_behaviors = [b for b in behaviors if time.time() - b['timestamp'] < 600]  # Last 10 minutes
            
            if not recent_behaviors:
                return 1.0
                
            # Count positive and negative behaviors
            positive_behaviors = sum(1 for b in recent_behaviors if b.get('positive', True))
            total_behaviors = len(recent_behaviors)
            
            base_trust = positive_behaviors / total_behaviors
            
            # Reduce trust based on suspicious patterns
            suspicious_count = self.suspicious_patterns.get(node_id, 0)
            trust_penalty = min(suspicious_count * 0.1, 0.5)  # Max 50% penalty
            
            return max(base_trust - trust_penalty, 0.0)
            
    def is_node_trusted(self, node_id: str) -> bool:
        """Check if a node is trusted"""
        return self.get_node_trust_score(node_id) > 0.5
        
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of Byzantine detection"""
        with self._lock:
            return {
                'monitored_nodes': len(self.node_behaviors),
                'suspicious_nodes': len([n for n in self.node_behaviors.keys() 
                                       if not self.is_node_trusted(n)]),
                'detection_threshold': self.detection_threshold,
                'total_behaviors_recorded': sum(len(behaviors) for behaviors in self.node_behaviors.values())
            }


class MessageSigner:
    """Handles message signing and verification"""
    
    def __init__(self, private_key: Optional[str] = None):
        self.private_key = private_key or secrets.token_hex(32)
        self.public_key = hashlib.sha256(self.private_key.encode()).hexdigest()
        
    def sign_message(self, message: ConsensusMessage) -> str:
        """Sign a consensus message"""
        # Create message digest
        message_data = {
            'message_id': message.message_id,
            'message_type': message.message_type.value,
            'sender_id': message.sender_id,
            'term': message.term,
            'proposal_id': message.proposal_id,
            'value': str(message.value),
            'timestamp': message.timestamp
        }
        
        message_json = json.dumps(message_data, sort_keys=True)
        signature = hmac.new(
            self.private_key.encode(),
            message_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    def verify_signature(self, message: ConsensusMessage, public_key: str) -> bool:
        """Verify message signature"""
        if not message.signature:
            return False
            
        try:
            # Recreate the message digest
            message_data = {
                'message_id': message.message_id,
                'message_type': message.message_type.value,
                'sender_id': message.sender_id,
                'term': message.term,
                'proposal_id': message.proposal_id,
                'value': str(message.value),
                'timestamp': message.timestamp
            }
            
            message_json = json.dumps(message_data, sort_keys=True)
            
            # For simplicity, we'll use a hash-based verification
            # In production, use proper digital signatures (RSA, ECDSA)
            expected_signature = hmac.new(
                hashlib.sha256(public_key.encode()).hexdigest().encode(),
                message_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(message.signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False


class PBFTConsensus:
    """Practical Byzantine Fault Tolerance consensus implementation"""
    
    def __init__(self, node_id: str, nodes: List[NodeInfo], 
                 byzantine_detector: Optional[ByzantineDetector] = None):
        self.node_id = node_id
        self.nodes = {node.node_id: node for node in nodes}
        self.byzantine_detector = byzantine_detector or ByzantineDetector()
        self.message_signer = MessageSigner()
        
        # Consensus state
        self.current_term = 0
        self.state = NodeState.FOLLOWER
        self.voted_for: Optional[str] = None
        self.leader_id: Optional[str] = None
        self.last_heartbeat = time.time()
        
        # Proposals and messages
        self.active_proposals: Dict[str, Proposal] = {}
        self.committed_values: List[Any] = []
        self.message_log: List[ConsensusMessage] = []
        
        # Configuration
        self.heartbeat_interval = 1.0
        self.election_timeout = 5.0
        self.message_timeout = 3.0
        
        # Callbacks
        self.on_value_committed: Optional[Callable[[Any], None]] = None
        self.on_leader_changed: Optional[Callable[[Optional[str]], None]] = None
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start the consensus node"""
        self._running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._election_timeout_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        logger.info(f"PBFT consensus node {self.node_id} started")
        
    async def stop(self):
        """Stop the consensus node"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info(f"PBFT consensus node {self.node_id} stopped")
        
    async def propose_value(self, value: Any) -> bool:
        """Propose a value for consensus"""
        if self.state != NodeState.LEADER:
            logger.warning(f"Only leader can propose values, current state: {self.state}")
            return False
            
        proposal = Proposal(
            proposer_id=self.node_id,
            value=value,
            term=self.current_term,
            required_majority=self._calculate_majority()
        )
        
        with self._lock:
            self.active_proposals[proposal.proposal_id] = proposal
            
        # Start PBFT phases
        success = await self._execute_pbft_phases(proposal)
        
        if success:
            logger.info(f"Value committed: {value}")
            if self.on_value_committed:
                self.on_value_committed(value)
                
        return success
        
    async def _execute_pbft_phases(self, proposal: Proposal) -> bool:
        """Execute the three phases of PBFT"""
        try:
            # Phase 1: Prepare
            if not await self._prepare_phase(proposal):
                return False
                
            # Phase 2: Accept
            if not await self._accept_phase(proposal):
                return False
                
            # Phase 3: Commit
            if not await self._commit_phase(proposal):
                return False
                
            # Value is committed
            with self._lock:
                self.committed_values.append(proposal.value)
                proposal.phase = ConsensusPhase.COMPLETED
                
            return True
            
        except Exception as e:
            logger.error(f"PBFT phases failed: {e}")
            with self._lock:
                proposal.phase = ConsensusPhase.FAILED
            return False
            
    async def _prepare_phase(self, proposal: Proposal) -> bool:
        """Execute prepare phase"""
        proposal.phase = ConsensusPhase.PREPARE_PHASE
        
        # Send prepare messages to all nodes
        prepare_message = ConsensusMessage(
            message_type=MessageType.PREPARE,
            sender_id=self.node_id,
            term=self.current_term,
            proposal_id=proposal.proposal_id,
            value=proposal.value
        )
        
        prepare_message.signature = self.message_signer.sign_message(prepare_message)
        
        # Broadcast to all nodes
        responses = await self._broadcast_message(prepare_message)
        
        # Count promises from trusted nodes
        promises = 0
        for response in responses:
            if (response.message_type == MessageType.PROMISE and
                response.proposal_id == proposal.proposal_id and
                self.byzantine_detector.is_node_trusted(response.sender_id)):
                
                proposal.promises.add(response.sender_id)
                promises += 1
                
        # Check if we have majority
        if promises >= proposal.required_majority:
            logger.info(f"Prepare phase successful for proposal {proposal.proposal_id}")
            return True
        else:
            logger.warning(f"Prepare phase failed for proposal {proposal.proposal_id}: {promises}/{proposal.required_majority}")
            return False
            
    async def _accept_phase(self, proposal: Proposal) -> bool:
        """Execute accept phase"""
        proposal.phase = ConsensusPhase.ACCEPT_PHASE
        
        # Send accept messages to all nodes
        accept_message = ConsensusMessage(
            message_type=MessageType.ACCEPT,
            sender_id=self.node_id,
            term=self.current_term,
            proposal_id=proposal.proposal_id,
            value=proposal.value
        )
        
        accept_message.signature = self.message_signer.sign_message(accept_message)
        
        # Broadcast to all nodes
        responses = await self._broadcast_message(accept_message)
        
        # Count accepts from trusted nodes
        accepts = 0
        for response in responses:
            if (response.message_type == MessageType.ACCEPTED and
                response.proposal_id == proposal.proposal_id and
                self.byzantine_detector.is_node_trusted(response.sender_id)):
                
                proposal.accepts.add(response.sender_id)
                accepts += 1
                
        # Check if we have majority
        if accepts >= proposal.required_majority:
            logger.info(f"Accept phase successful for proposal {proposal.proposal_id}")
            return True
        else:
            logger.warning(f"Accept phase failed for proposal {proposal.proposal_id}: {accepts}/{proposal.required_majority}")
            return False
            
    async def _commit_phase(self, proposal: Proposal) -> bool:
        """Execute commit phase"""
        proposal.phase = ConsensusPhase.COMMIT_PHASE
        
        # Send commit messages to all nodes
        commit_message = ConsensusMessage(
            message_type=MessageType.COMMIT,
            sender_id=self.node_id,
            term=self.current_term,
            proposal_id=proposal.proposal_id,
            value=proposal.value
        )
        
        commit_message.signature = self.message_signer.sign_message(commit_message)
        
        # Broadcast to all nodes
        responses = await self._broadcast_message(commit_message)
        
        # Count commits from trusted nodes
        commits = 0
        for response in responses:
            if (response.message_type == MessageType.COMMIT and
                response.proposal_id == proposal.proposal_id and
                self.byzantine_detector.is_node_trusted(response.sender_id)):
                
                proposal.commits.add(response.sender_id)
                commits += 1
                
        # Check if we have majority
        if commits >= proposal.required_majority:
            logger.info(f"Commit phase successful for proposal {proposal.proposal_id}")
            return True
        else:
            logger.warning(f"Commit phase failed for proposal {proposal.proposal_id}: {commits}/{proposal.required_majority}")
            return False
            
    async def _broadcast_message(self, message: ConsensusMessage) -> List[ConsensusMessage]:
        """Broadcast message to all nodes and collect responses"""
        responses = []
        
        # In a real implementation, this would send over network
        # For simulation, we'll just log the message
        logger.debug(f"Broadcasting {message.message_type.value} from {self.node_id}")
        
        # Simulate responses from other nodes
        for node_id, node_info in self.nodes.items():
            if node_id != self.node_id and self.byzantine_detector.is_node_trusted(node_id):
                # Simulate response based on message type
                response = await self._simulate_node_response(node_id, message)
                if response:
                    responses.append(response)
                    
        return responses
        
    async def _simulate_node_response(self, node_id: str, message: ConsensusMessage) -> Optional[ConsensusMessage]:
        """Simulate response from another node"""
        # This is a simulation - in real implementation, nodes would respond over network
        
        if message.message_type == MessageType.PREPARE:
            return ConsensusMessage(
                message_type=MessageType.PROMISE,
                sender_id=node_id,
                term=self.current_term,
                proposal_id=message.proposal_id
            )
        elif message.message_type == MessageType.ACCEPT:
            return ConsensusMessage(
                message_type=MessageType.ACCEPTED,
                sender_id=node_id,
                term=self.current_term,
                proposal_id=message.proposal_id
            )
        elif message.message_type == MessageType.COMMIT:
            return ConsensusMessage(
                message_type=MessageType.COMMIT,
                sender_id=node_id,
                term=self.current_term,
                proposal_id=message.proposal_id
            )
            
        return None
        
    def _calculate_majority(self) -> int:
        """Calculate required majority for Byzantine fault tolerance"""
        # For Byzantine fault tolerance, we need 2f+1 nodes where f is max faulty nodes
        # Majority is (2f+1)/2 + 1 = f+1
        trusted_nodes = sum(1 for node_id in self.nodes.keys() 
                          if self.byzantine_detector.is_node_trusted(node_id))
        
        # Assume up to 1/3 of nodes can be Byzantine
        max_faulty = trusted_nodes // 3
        return max_faulty + 1
        
    async def _heartbeat_loop(self):
        """Send periodic heartbeats when leader"""
        while self._running:
            try:
                if self.state == NodeState.LEADER:
                    await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                
    async def _send_heartbeat(self):
        """Send heartbeat to all followers"""
        heartbeat = ConsensusMessage(
            message_type=MessageType.HEARTBEAT,
            sender_id=self.node_id,
            term=self.current_term
        )
        
        heartbeat.signature = self.message_signer.sign_message(heartbeat)
        await self._broadcast_message(heartbeat)
        
    async def _election_timeout_loop(self):
        """Handle election timeouts"""
        while self._running:
            try:
                if (self.state == NodeState.FOLLOWER and 
                    time.time() - self.last_heartbeat > self.election_timeout):
                    await self._start_election()
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Election timeout loop error: {e}")
                
    async def _start_election(self):
        """Start leader election"""
        with self._lock:
            self.current_term += 1
            self.state = NodeState.CANDIDATE
            self.voted_for = self.node_id
            
        logger.info(f"Starting election for term {self.current_term}")
        
        # Send vote requests
        vote_request = ConsensusMessage(
            message_type=MessageType.VOTE_REQUEST,
            sender_id=self.node_id,
            term=self.current_term
        )
        
        vote_request.signature = self.message_signer.sign_message(vote_request)
        responses = await self._broadcast_message(vote_request)
        
        # Count votes
        votes = 1  # Vote for self
        for response in responses:
            if (response.message_type == MessageType.VOTE_RESPONSE and
                response.term == self.current_term and
                self.byzantine_detector.is_node_trusted(response.sender_id)):
                votes += 1
                
        # Check if won election
        required_votes = self._calculate_majority()
        if votes >= required_votes:
            with self._lock:
                self.state = NodeState.LEADER
                self.leader_id = self.node_id
                
            logger.info(f"Won election for term {self.current_term}")
            
            if self.on_leader_changed:
                self.on_leader_changed(self.node_id)
        else:
            with self._lock:
                self.state = NodeState.FOLLOWER
                
            logger.info(f"Lost election for term {self.current_term}: {votes}/{required_votes}")
            
    async def _cleanup_loop(self):
        """Clean up old proposals and messages"""
        while self._running:
            try:
                current_time = time.time()
                
                # Clean up old proposals
                with self._lock:
                    expired_proposals = [
                        pid for pid, proposal in self.active_proposals.items()
                        if current_time - proposal.timestamp > 300  # 5 minutes
                    ]
                    
                    for pid in expired_proposals:
                        del self.active_proposals[pid]
                        
                    # Clean up old messages
                    if len(self.message_log) > 10000:
                        self.message_log = self.message_log[-5000:]
                        
                await asyncio.sleep(60.0)  # Clean up every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                
    def get_consensus_status(self) -> Dict[str, Any]:
        """Get current consensus status"""
        with self._lock:
            trusted_nodes = [node_id for node_id in self.nodes.keys() 
                           if self.byzantine_detector.is_node_trusted(node_id)]
            
            return {
                'node_id': self.node_id,
                'state': self.state.value,
                'term': self.current_term,
                'leader_id': self.leader_id,
                'total_nodes': len(self.nodes),
                'trusted_nodes': len(trusted_nodes),
                'active_proposals': len(self.active_proposals),
                'committed_values': len(self.committed_values),
                'byzantine_detection': self.byzantine_detector.get_detection_summary()
            }


class ByzantineFaultToleranceManager:
    """Manages Byzantine fault tolerance across the system"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.consensus_nodes: Dict[str, PBFTConsensus] = {}
        self.byzantine_detector = ByzantineDetector()
        self.message_signer = MessageSigner()
        self._running = False
        
    async def start(self):
        """Start Byzantine fault tolerance system"""
        self._running = True
        
        # Start all consensus nodes
        for consensus_node in self.consensus_nodes.values():
            await consensus_node.start()
            
        logger.info("Byzantine fault tolerance system started")
        
    async def stop(self):
        """Stop Byzantine fault tolerance system"""
        self._running = False
        
        # Stop all consensus nodes
        for consensus_node in self.consensus_nodes.values():
            await consensus_node.stop()
            
        logger.info("Byzantine fault tolerance system stopped")
        
    def create_consensus_group(self, group_id: str, nodes: List[NodeInfo]) -> PBFTConsensus:
        """Create a new consensus group"""
        consensus_node = PBFTConsensus(
            node_id=self.node_id,
            nodes=nodes,
            byzantine_detector=self.byzantine_detector
        )
        
        self.consensus_nodes[group_id] = consensus_node
        logger.info(f"Created consensus group: {group_id} with {len(nodes)} nodes")
        
        return consensus_node
        
    async def propose_to_group(self, group_id: str, value: Any) -> bool:
        """Propose a value to a specific consensus group"""
        consensus_node = self.consensus_nodes.get(group_id)
        if not consensus_node:
            logger.error(f"Consensus group not found: {group_id}")
            return False
            
        return await consensus_node.propose_value(value)
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            'node_id': self.node_id,
            'running': self._running,
            'consensus_groups': len(self.consensus_nodes),
            'byzantine_detection': self.byzantine_detector.get_detection_summary(),
            'groups': {}
        }
        
        for group_id, consensus_node in self.consensus_nodes.items():
            status['groups'][group_id] = consensus_node.get_consensus_status()
            
        return status


# Utility functions for creating common configurations
def create_test_nodes(count: int) -> List[NodeInfo]:
    """Create test nodes for development"""
    nodes = []
    for i in range(count):
        node = NodeInfo(
            node_id=f"node_{i}",
            address=f"127.0.0.1",
            port=8000 + i
        )
        nodes.append(node)
    return nodes


def create_production_nodes(node_configs: List[Dict[str, Any]]) -> List[NodeInfo]:
    """Create production nodes from configuration"""
    nodes = []
    for config in node_configs:
        node = NodeInfo(
            node_id=config['node_id'],
            address=config['address'],
            port=config['port'],
            public_key=config.get('public_key')
        )
        nodes.append(node)
    return nodes