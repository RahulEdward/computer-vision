"""
Federated Learning Module for Computer Genie
फेडरेटेड लर्निंग मॉड्यूल - Computer Genie के लिए

This module implements privacy-preserving federated learning capabilities
for Computer Genie, allowing the system to learn from distributed user data
without compromising privacy.

Key Features:
- Privacy-preserving model updates (गोपनीयता-संरक्षित मॉडल अपडेट)
- Secure aggregation protocols (सुरक्षित एकत्रीकरण प्रोटोकॉल)
- Differential privacy mechanisms (विभेदक गोपनीयता तंत्र)
- Client selection strategies (क्लाइंट चयन रणनीतियां)
- Communication optimization (संचार अनुकूलन)
- Model compression for efficient transfer (कुशल स्थानांतरण के लिए मॉडल संपीड़न)
- Byzantine fault tolerance (बीजान्टाइन दोष सहनशीलता)
- Personalized federated learning (व्यक्तिगत फेडरेटेड लर्निंग)
- Cross-device and cross-silo scenarios (क्रॉस-डिवाइस और क्रॉस-साइलो परिदृश्य)
- Adaptive learning rate scheduling (अनुकूली शिक्षण दर निर्धारण)

Components:
- federated_server: Central coordination server (केंद्रीय समन्वय सर्वर)
- federated_client: Client-side training logic (क्लाइंट-साइड प्रशिक्षण तर्क)
- privacy_mechanisms: Differential privacy and secure aggregation (विभेदक गोपनीयता और सुरक्षित एकत्रीकरण)
- communication_manager: Efficient model transfer protocols (कुशल मॉडल स्थानांतरण प्रोटोकॉल)
- aggregation_strategies: Various aggregation algorithms (विभिन्न एकत्रीकरण एल्गोरिदम)
- client_selection: Smart client selection for training rounds (प्रशिक्षण राउंड के लिए स्मार्ट क्लाइंट चयन)
"""

from .federated_server import (
    FederatedServer,
    ServerConfig,
    AggregationStrategy,
    ClientSelectionStrategy
)

from .federated_client import (
    FederatedClient,
    ClientConfig,
    LocalTrainingConfig
)

from .privacy_mechanisms import (
    DifferentialPrivacy,
    SecureAggregation,
    PrivacyConfig,
    NoiseType
)

from .communication_manager import (
    CommunicationManager,
    CompressionType,
    CommunicationConfig
)

from .aggregation_strategies import (
    FedAvg,
    FedProx,
    FedNova,
    SCAFFOLD,
    AggregationResult
)

from .client_selection import (
    ClientSelector,
    SelectionStrategy,
    ClientMetrics
)

__all__ = [
    # Server components
    "FederatedServer",
    "ServerConfig", 
    "AggregationStrategy",
    "ClientSelectionStrategy",
    
    # Client components
    "FederatedClient",
    "ClientConfig",
    "LocalTrainingConfig",
    
    # Privacy mechanisms
    "DifferentialPrivacy",
    "SecureAggregation", 
    "PrivacyConfig",
    "NoiseType",
    
    # Communication
    "CommunicationManager",
    "CompressionType",
    "CommunicationConfig",
    
    # Aggregation
    "FedAvg",
    "FedProx", 
    "FedNova",
    "SCAFFOLD",
    "AggregationResult",
    
    # Client selection
    "ClientSelector",
    "SelectionStrategy",
    "ClientMetrics"
]

__version__ = "1.0.0"
__author__ = "Computer Genie AI Team"