"""
API and WebSocket Integration System
Provides standardized API responses and real-time WebSocket communication.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class ResponseStatus(Enum):
    """API response status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class MessageType(Enum):
    """WebSocket message type enumeration"""
    PROGRESS_UPDATE = "progress_update"
    STATUS_CHANGE = "status_change"
    ERROR_NOTIFICATION = "error_notification"
    COMPLETION_NOTIFICATION = "completion_notification"
    DATA_UPDATE = "data_update"
    HEARTBEAT = "heartbeat"

@dataclass
class APIResponse:
    """Standardized API response structure"""
    status: ResponseStatus
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    request_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    message_type: MessageType
    data: Dict[str, Any]
    timestamp: datetime = None
    message_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())

class ConnectionManager:
    """
    Manages WebSocket connections and broadcasting.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = {}  # user_id -> connection_ids
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, 
                     user_id: Optional[str] = None) -> None:
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        # Store connection metadata
        self.connection_metadata[connection_id] = {
            'user_id': user_id,
            'connected_at': datetime.utcnow(),
            'last_activity': datetime.utcnow()
        }
        
        # Add to user connections
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
    
    def disconnect(self, connection_id: str) -> None:
        """Remove WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remove from user connections
        metadata = self.connection_metadata.get(connection_id, {})
        user_id = metadata.get('user_id')
        if user_id and user_id in self.user_connections:
            if connection_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove metadata
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: WebSocketMessage, connection_id: str) -> bool:
        """Send message to specific connection"""
        if connection_id not in self.active_connections:
            return False
        
        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(asdict(message), default=str))
            
            # Update last activity
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]['last_activity'] = datetime.utcnow()
            
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.disconnect(connection_id)
            return False
    
    async def send_to_user(self, message: WebSocketMessage, user_id: str) -> int:
        """Send message to all connections for a user"""
        if user_id not in self.user_connections:
            return 0
        
        sent_count = 0
        for connection_id in self.user_connections[user_id].copy():
            if await self.send_personal_message(message, connection_id):
                sent_count += 1
        
        return sent_count
    
    async def broadcast(self, message: WebSocketMessage) -> int:
        """Broadcast message to all connections"""
        sent_count = 0
        for connection_id in list(self.active_connections.keys()):
            if await self.send_personal_message(message, connection_id):
                sent_count += 1
        
        return sent_count
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': len(self.active_connections),
            'users_with_connections': len(self.user_connections),
            'connections_per_user': {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            }
        }

class APIResponseBuilder:
    """
    Builder for standardized API responses.
    """
    
    @staticmethod
    def success(message: str, data: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Create success response"""
        return APIResponse(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=data,
            metadata=metadata
        )
    
    @staticmethod
    def error(message: str, errors: Optional[List[str]] = None,
              data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Create error response"""
        return APIResponse(
            status=ResponseStatus.ERROR,
            message=message,
            errors=errors or [message],
            data=data
        )
    
    @staticmethod
    def warning(message: str, warnings: Optional[List[str]] = None,
                data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Create warning response"""
        return APIResponse(
            status=ResponseStatus.WARNING,
            message=message,
            warnings=warnings or [message],
            data=data
        )
    
    @staticmethod
    def info(message: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Create info response"""
        return APIResponse(
            status=ResponseStatus.INFO,
            message=message,
            data=data
        )

class WebSocketMessageBuilder:
    """
    Builder for WebSocket messages.
    """
    
    @staticmethod
    def progress_update(progress: float, message: str, 
                       details: Optional[Dict[str, Any]] = None) -> WebSocketMessage:
        """Create progress update message"""
        data = {
            'progress': progress,
            'message': message,
            'details': details or {}
        }
        return WebSocketMessage(
            message_type=MessageType.PROGRESS_UPDATE,
            data=data
        )
    
    @staticmethod
    def status_change(status: str, message: str,
                     details: Optional[Dict[str, Any]] = None) -> WebSocketMessage:
        """Create status change message"""
        data = {
            'status': status,
            'message': message,
            'details': details or {}
        }
        return WebSocketMessage(
            message_type=MessageType.STATUS_CHANGE,
            data=data
        )
    
    @staticmethod
    def error_notification(error_message: str, error_code: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None) -> WebSocketMessage:
        """Create error notification message"""
        data = {
            'error_message': error_message,
            'error_code': error_code,
            'details': details or {}
        }
        return WebSocketMessage(
            message_type=MessageType.ERROR_NOTIFICATION,
            data=data
        )
    
    @staticmethod
    def completion_notification(message: str, results: Optional[Dict[str, Any]] = None) -> WebSocketMessage:
        """Create completion notification message"""
        data = {
            'message': message,
            'results': results or {}
        }
        return WebSocketMessage(
            message_type=MessageType.COMPLETION_NOTIFICATION,
            data=data
        )
    
    @staticmethod
    def data_update(data: Dict[str, Any], update_type: str = "general") -> WebSocketMessage:
        """Create data update message"""
        return WebSocketMessage(
            message_type=MessageType.DATA_UPDATE,
            data={
                'update_type': update_type,
                'data': data
            }
        )
    
    @staticmethod
    def heartbeat() -> WebSocketMessage:
        """Create heartbeat message"""
        return WebSocketMessage(
            message_type=MessageType.HEARTBEAT,
            data={'timestamp': datetime.utcnow().isoformat()}
        )

class RealTimeNotifier:
    """
    Real-time notification system for processing updates.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.notification_handlers: Dict[str, Callable] = {}
    
    async def notify_enrichment_progress(self, user_id: str, job_id: str, 
                                       progress: float, message: str) -> None:
        """Notify enrichment progress"""
        ws_message = WebSocketMessageBuilder.progress_update(
            progress=progress,
            message=message,
            details={'job_id': job_id, 'operation': 'enrichment'}
        )
        
        await self.connection_manager.send_to_user(ws_message, user_id)
    
    async def notify_document_analysis_progress(self, user_id: str, job_id: str,
                                              progress: float, message: str) -> None:
        """Notify document analysis progress"""
        ws_message = WebSocketMessageBuilder.progress_update(
            progress=progress,
            message=message,
            details={'job_id': job_id, 'operation': 'document_analysis'}
        )
        
        await self.connection_manager.send_to_user(ws_message, user_id)
    
    async def notify_processing_completion(self, user_id: str, job_id: str,
                                         results: Dict[str, Any]) -> None:
        """Notify processing completion"""
        ws_message = WebSocketMessageBuilder.completion_notification(
            message="Processing completed successfully",
            results=results
        )
        
        await self.connection_manager.send_to_user(ws_message, user_id)
    
    async def notify_processing_error(self, user_id: str, job_id: str,
                                    error_message: str, error_code: str = None) -> None:
        """Notify processing error"""
        ws_message = WebSocketMessageBuilder.error_notification(
            error_message=error_message,
            error_code=error_code,
            details={'job_id': job_id}
        )
        
        await self.connection_manager.send_to_user(ws_message, user_id)
    
    async def notify_data_update(self, user_id: str, update_type: str,
                               data: Dict[str, Any]) -> None:
        """Notify data update"""
        ws_message = WebSocketMessageBuilder.data_update(
            data=data,
            update_type=update_type
        )
        
        await self.connection_manager.send_to_user(ws_message, user_id)

class APIWebSocketIntegration:
    """
    Main integration system for API and WebSocket functionality.
    """
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.real_time_notifier = RealTimeNotifier(self.connection_manager)
        self.response_builder = APIResponseBuilder()
        self.message_builder = WebSocketMessageBuilder()
    
    def create_success_response(self, message: str, data: Optional[Dict[str, Any]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> JSONResponse:
        """Create and return success response"""
        response = self.response_builder.success(message, data, metadata)
        # Convert enum to string and datetime to ISO string for JSON serialization
        response_dict = asdict(response)
        response_dict['status'] = response.status.value
        response_dict['timestamp'] = response.timestamp.isoformat()
        return JSONResponse(
            status_code=200,
            content=response_dict
        )
    
    def create_error_response(self, message: str, errors: Optional[List[str]] = None,
                             status_code: int = 400) -> JSONResponse:
        """Create and return error response"""
        response = self.response_builder.error(message, errors)
        # Convert enum to string and datetime to ISO string for JSON serialization
        response_dict = asdict(response)
        response_dict['status'] = response.status.value
        response_dict['timestamp'] = response.timestamp.isoformat()
        return JSONResponse(
            status_code=status_code,
            content=response_dict
        )
    
    def create_warning_response(self, message: str, warnings: Optional[List[str]] = None,
                               data: Optional[Dict[str, Any]] = None) -> JSONResponse:
        """Create and return warning response"""
        response = self.response_builder.warning(message, warnings, data)
        # Convert enum to string and datetime to ISO string for JSON serialization
        response_dict = asdict(response)
        response_dict['status'] = response.status.value
        response_dict['timestamp'] = response.timestamp.isoformat()
        return JSONResponse(
            status_code=200,
            content=response_dict
        )
    
    async def handle_websocket_connection(self, websocket: WebSocket, 
                                        user_id: Optional[str] = None) -> None:
        """Handle WebSocket connection"""
        connection_id = str(uuid.uuid4())
        
        try:
            await self.connection_manager.connect(websocket, connection_id, user_id)
            
            # Send welcome message
            welcome_message = self.message_builder.info(
                message="Connected to real-time updates",
                data={'connection_id': connection_id}
            )
            await self.connection_manager.send_personal_message(welcome_message, connection_id)
            
            # Keep connection alive and handle messages
            while True:
                try:
                    # Wait for messages from client
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    # Handle different message types
                    await self._handle_websocket_message(connection_id, message_data)
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"WebSocket error for {connection_id}: {e}")
                    break
        
        finally:
            self.connection_manager.disconnect(connection_id)
    
    async def _handle_websocket_message(self, connection_id: str, 
                                      message_data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message"""
        message_type = message_data.get('type', 'unknown')
        
        if message_type == 'ping':
            # Respond to ping with pong
            pong_message = self.message_builder.heartbeat()
            await self.connection_manager.send_personal_message(pong_message, connection_id)
        
        elif message_type == 'subscribe':
            # Handle subscription requests
            subscription_type = message_data.get('subscription_type')
            await self._handle_subscription(connection_id, subscription_type)
        
        elif message_type == 'unsubscribe':
            # Handle unsubscription requests
            subscription_type = message_data.get('subscription_type')
            await self._handle_unsubscription(connection_id, subscription_type)
    
    async def _handle_subscription(self, connection_id: str, subscription_type: str) -> None:
        """Handle subscription to specific updates"""
        # In a real implementation, this would manage subscriptions
        # For now, we'll just acknowledge the subscription
        ack_message = self.message_builder.info(
            message=f"Subscribed to {subscription_type} updates",
            data={'subscription_type': subscription_type}
        )
        await self.connection_manager.send_personal_message(ack_message, connection_id)
    
    async def _handle_unsubscription(self, connection_id: str, subscription_type: str) -> None:
        """Handle unsubscription from specific updates"""
        # In a real implementation, this would manage unsubscriptions
        # For now, we'll just acknowledge the unsubscription
        ack_message = self.message_builder.info(
            message=f"Unsubscribed from {subscription_type} updates",
            data={'subscription_type': subscription_type}
        )
        await self.connection_manager.send_personal_message(ack_message, connection_id)
    
    async def broadcast_system_status(self, status: str, message: str) -> None:
        """Broadcast system status to all connected users"""
        status_message = self.message_builder.status_change(status, message)
        await self.connection_manager.broadcast(status_message)
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        connection_stats = self.connection_manager.get_connection_stats()
        
        return {
            'websocket_connections': connection_stats,
            'api_responses_sent': 0,  # Would track in production
            'websocket_messages_sent': 0,  # Would track in production
            'active_notifications': 0  # Would track in production
        }

# Global integration system instance
_global_integration_system: Optional[APIWebSocketIntegration] = None

def get_global_integration_system() -> APIWebSocketIntegration:
    """Get or create global integration system"""
    global _global_integration_system
    
    if _global_integration_system is None:
        _global_integration_system = APIWebSocketIntegration()
    
    return _global_integration_system
