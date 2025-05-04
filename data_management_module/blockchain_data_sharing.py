"""
Blockchain Data Sharing Module

This module implements a blockchain-based data sharing system for pet health data.
It ensures data integrity, secure sharing, and transparent data access logs.

Technical Note: This implementation is based on patented technology filed by Ucaretron Inc.
"""

import time
import json
import os
import hashlib
import uuid
import base64
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('blockchain_data_sharing')

class Block:
    """
    Block class for blockchain implementation.
    
    Each block contains pet health data records, access logs, or sharing permissions
    that have been validated and added to the blockchain.
    """
    
    def __init__(self, index: int, timestamp: float, data: Dict[str, Any], 
                 previous_hash: str, nonce: int = 0):
        """
        Initialize a new block.
        
        Args:
            index: Block index in the chain
            timestamp: Block creation time
            data: Data to be stored in the block
            previous_hash: Hash of the previous block
            nonce: Value used for mining proof-of-work
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """
        Calculate the hash of the block.
        
        Returns:
            SHA-256 hash of the block
        """
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """
        Mine the block with proof-of-work.
        
        Args:
            difficulty: Number of leading zeros required in hash
        """
        target = '0' * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary.
        
        Returns:
            Dictionary representation of the block
        """
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }
    
    @classmethod
    def from_dict(cls, block_dict: Dict[str, Any]) -> 'Block':
        """
        Create a block from dictionary.
        
        Args:
            block_dict: Dictionary representation of a block
            
        Returns:
            Block instance
        """
        block = cls(
            block_dict['index'],
            block_dict['timestamp'],
            block_dict['data'],
            block_dict['previous_hash'],
            block_dict['nonce']
        )
        block.hash = block_dict['hash']
        return block


class PetDataBlockchain:
    """
    Blockchain implementation for pet health data.
    
    This class manages the blockchain, adding new blocks for data records,
    access logs, and sharing permissions.
    """
    
    def __init__(self, chain_directory: str = './blockchain_data', difficulty: int = 4):
        """
        Initialize the blockchain.
        
        Args:
            chain_directory: Directory to store blockchain data
            difficulty: Mining difficulty (number of leading zeros required in hash)
        """
        self.chain_directory = chain_directory
        self.difficulty = difficulty
        self.chain = []
        self.pending_transactions = []
        self.mining_lock = threading.Lock()
        
        # Create directory if it doesn't exist
        if not os.path.exists(chain_directory):
            os.makedirs(chain_directory)
        
        # Load existing chain or create genesis block
        self._load_chain()
        
        if not self.chain:
            self._create_genesis_block()
            self._save_chain()
        
        logger.info(f"Blockchain initialized with {len(self.chain)} blocks")
    
    def _create_genesis_block(self) -> None:
        """Create the genesis block."""
        genesis_block = Block(0, time.time(), {
            'message': 'Genesis Block',
            'system': 'Pet Health Monitoring System',
            'created_at': datetime.now().isoformat()
        }, '0')
        self.chain.append(genesis_block)
        logger.info("Genesis block created")
    
    def _save_chain(self) -> None:
        """Save blockchain to disk."""
        chain_data = [block.to_dict() for block in self.chain]
        file_path = os.path.join(self.chain_directory, 'blockchain.json')
        
        with open(file_path, 'w') as f:
            json.dump(chain_data, f, indent=2)
        
        # Save each block individually for redundancy
        blocks_dir = os.path.join(self.chain_directory, 'blocks')
        if not os.path.exists(blocks_dir):
            os.makedirs(blocks_dir)
        
        for block in self.chain:
            block_file = os.path.join(blocks_dir, f"block_{block.index}.json")
            with open(block_file, 'w') as f:
                json.dump(block.to_dict(), f, indent=2)
    
    def _load_chain(self) -> None:
        """Load blockchain from disk."""
        file_path = os.path.join(self.chain_directory, 'blockchain.json')
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    chain_data = json.load(f)
                
                self.chain = [Block.from_dict(block_dict) for block_dict in chain_data]
                logger.info(f"Loaded blockchain with {len(self.chain)} blocks")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading blockchain: {str(e)}")
                # In case of error, try to recover from individual block files
                self._recover_from_block_files()
    
    def _recover_from_block_files(self) -> None:
        """Recover blockchain from individual block files."""
        blocks_dir = os.path.join(self.chain_directory, 'blocks')
        
        if not os.path.exists(blocks_dir):
            logger.warning("No block files found for recovery")
            return
        
        # Get all block files
        block_files = []
        for filename in os.listdir(blocks_dir):
            if filename.startswith('block_') and filename.endswith('.json'):
                try:
                    index = int(filename.split('_')[1].split('.')[0])
                    block_files.append((index, filename))
                except ValueError:
                    continue
        
        # Sort by index
        block_files.sort(key=lambda x: x[0])
        
        # Load blocks
        recovered_chain = []
        for index, filename in block_files:
            file_path = os.path.join(blocks_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    block_dict = json.load(f)
                block = Block.from_dict(block_dict)
                recovered_chain.append(block)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading block file {filename}: {str(e)}")
        
        # Verify recovered chain
        if self._is_chain_valid(recovered_chain):
            self.chain = recovered_chain
            logger.info(f"Recovered blockchain with {len(self.chain)} blocks")
        else:
            logger.error("Recovered chain is invalid")
            if recovered_chain:
                # Start with genesis block
                if recovered_chain[0].index == 0:
                    self.chain = [recovered_chain[0]]
                    logger.info("Recovered genesis block only")
    
    def get_latest_block(self) -> Block:
        """
        Get the latest block in the chain.
        
        Returns:
            Latest block
        """
        return self.chain[-1]
    
    def add_block(self, data: Dict[str, Any]) -> Block:
        """
        Add a new block to the chain.
        
        Args:
            data: Data to store in the block
            
        Returns:
            Newly created block
        """
        with self.mining_lock:
            previous_block = self.get_latest_block()
            new_index = previous_block.index + 1
            new_block = Block(
                new_index,
                time.time(),
                data,
                previous_block.hash
            )
            
            # Mine the block
            new_block.mine_block(self.difficulty)
            
            # Add to chain
            self.chain.append(new_block)
            
            # Save to disk
            self._save_chain()
            
            logger.info(f"Added new block with index {new_index}")
            return new_block
    
    def add_transaction(self, data: Dict[str, Any]) -> int:
        """
        Add a transaction to pending transactions.
        
        Args:
            data: Transaction data
            
        Returns:
            Index of transaction in pending transactions
        """
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
        
        # Add transaction ID if not present
        if 'transaction_id' not in data:
            data['transaction_id'] = str(uuid.uuid4())
        
        self.pending_transactions.append(data)
        idx = len(self.pending_transactions) - 1
        logger.info(f"Added transaction to pool (ID: {data['transaction_id']})")
        return idx
    
    def mine_pending_transactions(self, mining_reward_address: str = None) -> Optional[Block]:
        """
        Mine pending transactions into a new block.
        
        Args:
            mining_reward_address: Address to receive mining reward (if reward system is used)
            
        Returns:
            Newly created block or None if no pending transactions
        """
        # If no pending transactions, do nothing
        if not self.pending_transactions:
            return None
        
        with self.mining_lock:
            # Create a new block with pending transactions
            block_data = {
                'transactions': self.pending_transactions,
                'mined_by': mining_reward_address if mining_reward_address else 'system'
            }
            
            # Add the block to the chain
            new_block = self.add_block(block_data)
            
            # Clear pending transactions
            self.pending_transactions = []
            
            logger.info(f"Mined {len(block_data['transactions'])} transactions into block {new_block.index}")
            return new_block
    
    def _is_chain_valid(self, chain: List[Block] = None) -> bool:
        """
        Validate the integrity of the blockchain.
        
        Args:
            chain: Chain to validate (uses self.chain if None)
            
        Returns:
            True if valid, False otherwise
        """
        if chain is None:
            chain = self.chain
        
        # Check if chain is empty
        if not chain:
            return False
        
        # Validate each block
        for i in range(1, len(chain)):
            current_block = chain[i]
            previous_block = chain[i - 1]
            
            # Validate block hash
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Block {current_block.index} has invalid hash")
                return False
            
            # Validate previous hash reference
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Block {current_block.index} has invalid previous hash")
                return False
        
        return True
    
    def validate_chain(self) -> bool:
        """
        Validate the integrity of the blockchain.
        
        Returns:
            True if valid, False otherwise
        """
        is_valid = self._is_chain_valid()
        if is_valid:
            logger.info("Blockchain validation successful")
        else:
            logger.error("Blockchain validation failed")
        return is_valid
    
    def get_chain_length(self) -> int:
        """
        Get the length of the blockchain.
        
        Returns:
            Number of blocks in the chain
        """
        return len(self.chain)
    
    def get_block_by_index(self, index: int) -> Optional[Block]:
        """
        Get a block by its index.
        
        Args:
            index: Block index
            
        Returns:
            Block if found, None otherwise
        """
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def get_block_by_hash(self, hash_value: str) -> Optional[Block]:
        """
        Get a block by its hash.
        
        Args:
            hash_value: Block hash
            
        Returns:
            Block if found, None otherwise
        """
        for block in self.chain:
            if block.hash == hash_value:
                return block
        return None
    
    def get_transactions_by_pet_id(self, pet_id: str) -> List[Dict[str, Any]]:
        """
        Get all transactions related to a specific pet.
        
        Args:
            pet_id: Pet ID to search for
            
        Returns:
            List of transactions
        """
        transactions = []
        
        for block in self.chain:
            # Skip genesis block
            if block.index == 0:
                continue
            
            if 'transactions' in block.data:
                for tx in block.data['transactions']:
                    # Check if transaction is related to this pet
                    if 'pet_id' in tx and tx['pet_id'] == pet_id:
                        # Add block info to transaction
                        tx_copy = tx.copy()
                        tx_copy['block_index'] = block.index
                        tx_copy['block_hash'] = block.hash
                        tx_copy['block_timestamp'] = block.timestamp
                        transactions.append(tx_copy)
        
        return transactions


class BlockchainDataSharing:
    """
    Blockchain-based data sharing system for pet health data.
    
    This class provides functionality for securely sharing pet health data
    using blockchain technology to ensure data integrity and maintain
    a transparent access log.
    """
    
    def __init__(self, data_directory: str = './blockchain_sharing'):
        """
        Initialize the blockchain data sharing system.
        
        Args:
            data_directory: Directory to store blockchain and shared data
        """
        self.data_directory = data_directory
        
        # Create directories if they don't exist
        self._create_directory_structure()
        
        # Initialize blockchain
        blockchain_dir = os.path.join(data_directory, 'blockchain')
        self.blockchain = PetDataBlockchain(blockchain_dir)
        
        # Initialize sharing settings and permissions
        self.sharing_settings = self._load_sharing_settings()
        
        logger.info("Blockchain Data Sharing module initialized")
    
    def _create_directory_structure(self) -> None:
        """Create the necessary directory structure for data storage."""
        # Create main data directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
        # Create subdirectories
        subdirs = ['blockchain', 'shared_data', 'encrypted_keys', 'permissions']
        for subdir in subdirs:
            path = os.path.join(self.data_directory, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def _load_sharing_settings(self) -> Dict[str, Any]:
        """
        Load sharing settings from disk.
        
        Returns:
            Sharing settings dictionary
        """
        settings_path = os.path.join(self.data_directory, 'sharing_settings.json')
        
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                logger.info("Loaded sharing settings")
                return settings
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading sharing settings: {str(e)}")
        
        # Default settings
        default_settings = {
            'auto_sync_enabled': True,
            'default_sharing_duration': 30 * 24 * 60 * 60,  # 30 days in seconds
            'default_encryption_level': 'AES-256',
            'access_log_enabled': True,
            'require_consent_confirmation': True,
            'data_categories': {
                'basic_profile': {
                    'description': 'Basic pet profile information',
                    'default_shared': True
                },
                'medical_records': {
                    'description': 'Medical examination and treatment records',
                    'default_shared': False
                },
                'sensor_data': {
                    'description': 'Raw sensor data from monitoring devices',
                    'default_shared': False
                },
                'behavior_analysis': {
                    'description': 'Behavior analysis and recommendations',
                    'default_shared': True
                },
                'care_plans': {
                    'description': 'Care plans and treatment schedules',
                    'default_shared': True
                }
            }
        }
        
        # Save default settings
        try:
            with open(settings_path, 'w') as f:
                json.dump(default_settings, f, indent=2)
            logger.info("Created default sharing settings")
        except IOError as e:
            logger.error(f"Error saving default sharing settings: {str(e)}")
        
        return default_settings
    
    def _save_sharing_settings(self) -> bool:
        """
        Save sharing settings to disk.
        
        Returns:
            True if successful, False otherwise
        """
        settings_path = os.path.join(self.data_directory, 'sharing_settings.json')
        
        try:
            with open(settings_path, 'w') as f:
                json.dump(self.sharing_settings, f, indent=2)
            logger.info("Saved sharing settings")
            return True
        except IOError as e:
            logger.error(f"Error saving sharing settings: {str(e)}")
            return False
    
    def update_sharing_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        Update sharing settings.
        
        Args:
            new_settings: New settings to merge with existing settings
            
        Returns:
            True if successful, False otherwise
        """
        # Update settings
        self.sharing_settings.update(new_settings)
        
        # Save updated settings
        return self._save_sharing_settings()
    
    def get_sharing_settings(self) -> Dict[str, Any]:
        """
        Get current sharing settings.
        
        Returns:
            Current sharing settings
        """
        return self.sharing_settings
    
    def share_pet_data(self, pet_id: str, recipient_id: str, data_categories: List[str] = None,
                      expiration_time: int = None, access_level: str = 'read',
                      custom_message: str = None) -> Dict[str, Any]:
        """
        Share pet data with a recipient.
        
        Args:
            pet_id: ID of the pet whose data is being shared
            recipient_id: ID of the recipient (veterinarian, researcher, etc.)
            data_categories: Categories of data to share (if None, use defaults)
            expiration_time: Time when sharing expires (in seconds since epoch)
            access_level: Access level ('read', 'write', 'admin')
            custom_message: Custom message to recipient
            
        Returns:
            Sharing information including status and share_id
        """
        # Use default data categories if none provided
        if data_categories is None:
            data_categories = [
                category for category, settings in self.sharing_settings['data_categories'].items()
                if settings.get('default_shared', False)
            ]
        
        # Use default sharing duration if expiration not provided
        if expiration_time is None:
            expiration_time = int(time.time() + self.sharing_settings['default_sharing_duration'])
        
        # Generate share ID
        share_id = f"share_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Create sharing record
        sharing_data = {
            'type': 'data_sharing',
            'share_id': share_id,
            'pet_id': pet_id,
            'granted_by': 'pet_owner',  # Assuming the owner is granting access
            'recipient_id': recipient_id,
            'data_categories': data_categories,
            'access_level': access_level,
            'created_at': time.time(),
            'expires_at': expiration_time,
            'custom_message': custom_message,
            'status': 'active'
        }
        
        # Add transaction to blockchain
        self.blockchain.add_transaction(sharing_data)
        
        # Mine transaction
        self.blockchain.mine_pending_transactions()
        
        # Save sharing permission to disk
        permissions_dir = os.path.join(self.data_directory, 'permissions')
        permission_file = os.path.join(permissions_dir, f"{share_id}.json")
        try:
            with open(permission_file, 'w') as f:
                json.dump(sharing_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving sharing permission: {str(e)}")
        
        logger.info(f"Shared pet {pet_id} data with recipient {recipient_id} (Share ID: {share_id})")
        
        return {
            'status': 'success',
            'share_id': share_id,
            'pet_id': pet_id,
            'recipient_id': recipient_id,
            'expires_at': expiration_time
        }
    
    def revoke_shared_data(self, share_id: str, reason: str = None) -> Dict[str, Any]:
        """
        Revoke previously shared data.
        
        Args:
            share_id: ID of the sharing record to revoke
            reason: Reason for revocation
            
        Returns:
            Result of revocation operation
        """
        # Check if sharing permission exists
        permissions_dir = os.path.join(self.data_directory, 'permissions')
        permission_file = os.path.join(permissions_dir, f"{share_id}.json")
        
        if not os.path.exists(permission_file):
            return {
                'status': 'error',
                'message': f"Sharing record {share_id} not found"
            }
        
        try:
            # Load sharing permission
            with open(permission_file, 'r') as f:
                sharing_data = json.load(f)
            
            # Update status to revoked
            sharing_data['status'] = 'revoked'
            sharing_data['revoked_at'] = time.time()
            sharing_data['revocation_reason'] = reason
            
            # Save updated permission
            with open(permission_file, 'w') as f:
                json.dump(sharing_data, f, indent=2)
            
            # Add revocation to blockchain
            revocation_data = {
                'type': 'data_sharing_revocation',
                'share_id': share_id,
                'revoked_at': time.time(),
                'revocation_reason': reason,
                'pet_id': sharing_data['pet_id'],
                'recipient_id': sharing_data['recipient_id']
            }
            
            self.blockchain.add_transaction(revocation_data)
            self.blockchain.mine_pending_transactions()
            
            logger.info(f"Revoked sharing permission {share_id}")
            
            return {
                'status': 'success',
                'message': f"Sharing permission {share_id} revoked successfully",
                'revoked_at': revocation_data['revoked_at']
            }
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error revoking sharing permission: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error revoking sharing permission: {str(e)}"
            }
    
    def validate_access_permission(self, pet_id: str, requester_id: str,
                                  data_category: str = None) -> Dict[str, Any]:
        """
        Validate if a requester has permission to access pet data.
        
        Args:
            pet_id: ID of the pet
            requester_id: ID of the requester
            data_category: Specific data category to validate access for
            
        Returns:
            Validation result
        """
        # Check all active permissions
        permissions_dir = os.path.join(self.data_directory, 'permissions')
        active_permissions = []
        
        if not os.path.exists(permissions_dir):
            return {
                'status': 'error',
                'has_access': False,
                'message': "Permissions directory not found"
            }
        
        current_time = time.time()
        
        for filename in os.listdir(permissions_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                file_path = os.path.join(permissions_dir, filename)
                with open(file_path, 'r') as f:
                    permission = json.load(f)
                
                # Check if permission is for this pet and requester
                if (permission['pet_id'] == pet_id and 
                    permission['recipient_id'] == requester_id and
                    permission['status'] == 'active'):
                    
                    # Check if permission is still valid
                    if permission['expires_at'] > current_time:
                        # Check if data category is included (if specified)
                        if data_category is None or data_category in permission['data_categories']:
                            active_permissions.append(permission)
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.error(f"Error reading permission file {filename}: {str(e)}")
        
        if active_permissions:
            # Sort by expiration time (most recent first)
            active_permissions.sort(key=lambda x: x['expires_at'], reverse=True)
            
            # Log access validation in blockchain
            access_log = {
                'type': 'access_validation',
                'pet_id': pet_id,
                'requester_id': requester_id,
                'data_category': data_category,
                'timestamp': current_time,
                'permission_granted': True,
                'share_id': active_permissions[0]['share_id']
            }
            
            if self.sharing_settings.get('access_log_enabled', True):
                self.blockchain.add_transaction(access_log)
                self.blockchain.mine_pending_transactions()
            
            return {
                'status': 'success',
                'has_access': True,
                'permissions': active_permissions,
                'message': "Access granted"
            }
        else:
            # Log access denial in blockchain
            access_log = {
                'type': 'access_validation',
                'pet_id': pet_id,
                'requester_id': requester_id,
                'data_category': data_category,
                'timestamp': current_time,
                'permission_granted': False
            }
            
            if self.sharing_settings.get('access_log_enabled', True):
                self.blockchain.add_transaction(access_log)
                self.blockchain.mine_pending_transactions()
            
            return {
                'status': 'success',
                'has_access': False,
                'message': "No active permissions found"
            }
    
    def log_data_access(self, pet_id: str, requester_id: str, data_category: str,
                        operation: str, result: str = 'success') -> Dict[str, Any]:
        """
        Log data access events to the blockchain.
        
        Args:
            pet_id: ID of the pet whose data is being accessed
            requester_id: ID of the entity accessing the data
            data_category: Category of data being accessed
            operation: Type of operation performed (read, write, etc.)
            result: Result of the operation
            
        Returns:
            Log operation result
        """
        if not self.sharing_settings.get('access_log_enabled', True):
            return {
                'status': 'skipped',
                'message': "Access logging is disabled"
            }
        
        log_data = {
            'type': 'data_access_log',
            'pet_id': pet_id,
            'requester_id': requester_id,
            'data_category': data_category,
            'operation': operation,
            'timestamp': time.time(),
            'result': result
        }
        
        # Add transaction to blockchain
        self.blockchain.add_transaction(log_data)
        
        # Mine transaction (can be done asynchronously)
        # For better performance, mining can be done periodically instead of after each transaction
        mining_thread = threading.Thread(target=self.blockchain.mine_pending_transactions)
        mining_thread.daemon = True  # Make thread daemon so it doesn't block application exit
        mining_thread.start()
        
        logger.info(f"Logged {operation} access to {data_category} data for pet {pet_id} by {requester_id}")
        
        return {
            'status': 'success',
            'message': "Access logged successfully"
        }
    
    def get_active_shares_for_pet(self, pet_id: str) -> List[Dict[str, Any]]:
        """
        Get all active sharing permissions for a pet.
        
        Args:
            pet_id: ID of the pet
            
        Returns:
            List of active sharing permissions
        """
        permissions_dir = os.path.join(self.data_directory, 'permissions')
        active_shares = []
        
        if not os.path.exists(permissions_dir):
            return active_shares
        
        current_time = time.time()
        
        for filename in os.listdir(permissions_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                file_path = os.path.join(permissions_dir, filename)
                with open(file_path, 'r') as f:
                    permission = json.load(f)
                
                # Check if permission is for this pet and still active
                if (permission['pet_id'] == pet_id and 
                    permission['status'] == 'active' and
                    permission['expires_at'] > current_time):
                    active_shares.append(permission)
            except (json.JSONDecodeError, IOError, KeyError) as e:
                logger.error(f"Error reading permission file {filename}: {str(e)}")
        
        # Sort by creation time (most recent first)
        active_shares.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        
        return active_shares
    
    def get_access_logs_for_pet(self, pet_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get access logs for a pet from the blockchain.
        
        Args:
            pet_id: ID of the pet
            limit: Maximum number of logs to return
            
        Returns:
            List of access logs
        """
        transactions = self.blockchain.get_transactions_by_pet_id(pet_id)
        
        # Filter for access logs and validations
        access_logs = [
            tx for tx in transactions
            if tx.get('type') in ['data_access_log', 'access_validation']
        ]
        
        # Sort by timestamp (most recent first)
        access_logs.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Limit number of logs
        return access_logs[:limit]
    
    def verify_data_integrity(self, pet_id: str, data_hash: str) -> Dict[str, Any]:
        """
        Verify the integrity of pet data using blockchain.
        
        Args:
            pet_id: ID of the pet
            data_hash: Hash of the data to verify
            
        Returns:
            Verification result
        """
        transactions = self.blockchain.get_transactions_by_pet_id(pet_id)
        
        # Filter for data integrity records
        integrity_records = [
            tx for tx in transactions
            if tx.get('type') == 'data_integrity' and 'data_hash' in tx
        ]
        
        # Check if data hash exists
        for record in integrity_records:
            if record['data_hash'] == data_hash:
                return {
                    'status': 'success',
                    'verified': True,
                    'message': "Data integrity verified",
                    'recorded_at': record['timestamp'],
                    'block_index': record['block_index']
                }
        
        return {
            'status': 'success',
            'verified': False,
            'message': "No matching data hash found in blockchain"
        }
    
    def register_data_integrity(self, pet_id: str, data_category: str, 
                               data_hash: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Register data integrity hash in the blockchain.
        
        Args:
            pet_id: ID of the pet
            data_category: Category of the data
            data_hash: Hash of the data
            metadata: Additional metadata about the data
            
        Returns:
            Registration result
        """
        if metadata is None:
            metadata = {}
        
        integrity_data = {
            'type': 'data_integrity',
            'pet_id': pet_id,
            'data_category': data_category,
            'data_hash': data_hash,
            'timestamp': time.time(),
            'metadata': metadata
        }
        
        # Add transaction to blockchain
        self.blockchain.add_transaction(integrity_data)
        
        # Mine transaction
        new_block = self.blockchain.mine_pending_transactions()
        
        logger.info(f"Registered integrity hash for {data_category} data of pet {pet_id}")
        
        return {
            'status': 'success',
            'message': "Data integrity hash registered",
            'block_index': new_block.index if new_block else None,
            'block_hash': new_block.hash if new_block else None
        }


class SharedDataExchange:
    """
    Handles exchange of shared data between systems.
    
    This class provides functionality for exporting and importing shared data
    using secure channels and blockchain verification.
    """
    
    def __init__(self, blockchain_sharing: BlockchainDataSharing, 
                encryption_key: str = None):
        """
        Initialize the shared data exchange.
        
        Args:
            blockchain_sharing: BlockchainDataSharing instance
            encryption_key: Key for encrypting shared data (if None, generate one)
        """
        self.blockchain_sharing = blockchain_sharing
        
        # Generate encryption key if not provided
        if encryption_key is None:
            encryption_key = self._generate_encryption_key()
        
        self.encryption_key = encryption_key
        logger.info("Shared Data Exchange initialized")
    
    def _generate_encryption_key(self) -> str:
        """
        Generate a new encryption key.
        
        Returns:
            Encryption key
        """
        key = base64.b64encode(os.urandom(32)).decode('utf-8')
        logger.info("Generated new encryption key")
        return key
    
    def _encrypt_data(self, data: Dict[str, Any]) -> str:
        """
        Encrypt data for sharing.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        # Convert data to JSON string
        data_str = json.dumps(data)
        
        # For demonstration, using simple XOR encryption
        # In production, use a proper encryption library like cryptography
        key_bytes = self.encryption_key.encode('utf-8')
        data_bytes = data_str.encode('utf-8')
        
        # XOR each byte with key (cycling through key)
        encrypted = bytes([
            data_bytes[i] ^ key_bytes[i % len(key_bytes)]
            for i in range(len(data_bytes))
        ])
        
        # Return as base64
        return base64.b64encode(encrypted).decode('utf-8')
    
    def _decrypt_data(self, encrypted_data: str, key: str = None) -> Dict[str, Any]:
        """
        Decrypt shared data.
        
        Args:
            encrypted_data: Encrypted data as base64 string
            key: Decryption key (uses self.encryption_key if None)
            
        Returns:
            Decrypted data
        """
        if key is None:
            key = self.encryption_key
        
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_data)
        key_bytes = key.encode('utf-8')
        
        # XOR each byte with key (cycling through key)
        decrypted_bytes = bytes([
            encrypted_bytes[i] ^ key_bytes[i % len(key_bytes)]
            for i in range(len(encrypted_bytes))
        ])
        
        # Parse JSON
        return json.loads(decrypted_bytes.decode('utf-8'))
    
    def export_shared_data(self, pet_id: str, recipient_id: str,
                          data_categories: List[str]) -> Dict[str, Any]:
        """
        Export shared data for a recipient.
        
        Args:
            pet_id: ID of the pet
            recipient_id: ID of the recipient
            data_categories: Categories of data to export
            
        Returns:
            Export result including secure package
        """
        # Validate recipient has access permission
        for category in data_categories:
            validation = self.blockchain_sharing.validate_access_permission(
                pet_id, recipient_id, category
            )
            
            if not validation['has_access']:
                return {
                    'status': 'error',
                    'message': f"Recipient does not have access to {category} data"
                }
        
        # Collect data to share
        shared_data = {
            'pet_id': pet_id,
            'recipient_id': recipient_id,
            'export_timestamp': time.time(),
            'data': {}
        }
        
        # TODO: Collect actual data from storage
        # This would involve fetching data from a storage manager
        # For demonstration, using dummy data
        for category in data_categories:
            shared_data['data'][category] = {
                'sample_data': f"Sample {category} data for pet {pet_id}",
                'last_updated': time.time()
            }
        
        # Generate data hash for integrity verification
        data_hash = hashlib.sha256(json.dumps(shared_data, sort_keys=True).encode()).hexdigest()
        shared_data['data_hash'] = data_hash
        
        # Register data integrity in blockchain
        self.blockchain_sharing.register_data_integrity(
            pet_id,
            'shared_data_export',
            data_hash,
            {
                'recipient_id': recipient_id,
                'categories': data_categories,
                'export_timestamp': shared_data['export_timestamp']
            }
        )
        
        # Log data access
        for category in data_categories:
            self.blockchain_sharing.log_data_access(
                pet_id,
                recipient_id,
                category,
                'export'
            )
        
        # Encrypt data
        encrypted_data = self._encrypt_data(shared_data)
        
        # Generate one-time password for recipient
        otp = str(uuid.uuid4())[:8]
        
        # Prepare secure package
        secure_package = {
            'encrypted_data': encrypted_data,
            'pet_id': pet_id,
            'recipient_id': recipient_id,
            'export_timestamp': shared_data['export_timestamp'],
            'data_categories': data_categories,
            'data_hash': data_hash,
            'verification_code': otp
        }
        
        logger.info(f"Exported shared data for pet {pet_id} to recipient {recipient_id}")
        
        return {
            'status': 'success',
            'secure_package': secure_package,
            'otp': otp,  # In production, this would be sent separately
            'message': "Data exported successfully"
        }
    
    def import_shared_data(self, secure_package: Dict[str, Any], otp: str,
                          decryption_key: str) -> Dict[str, Any]:
        """
        Import shared data from a secure package.
        
        Args:
            secure_package: Secure package from export_shared_data
            otp: One-time password for verification
            decryption_key: Key for decrypting data
            
        Returns:
            Import result including decrypted data
        """
        # Verify OTP
        if secure_package.get('verification_code') != otp:
            return {
                'status': 'error',
                'message': "Invalid verification code"
            }
        
        # Decrypt data
        try:
            decrypted_data = self._decrypt_data(
                secure_package['encrypted_data'],
                decryption_key
            )
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            return {
                'status': 'error',
                'message': f"Error decrypting data: {str(e)}"
            }
        
        # Verify data hash
        computed_hash = hashlib.sha256(
            json.dumps({k: v for k, v in decrypted_data.items() if k != 'data_hash'}, sort_keys=True).encode()
        ).hexdigest()
        
        if computed_hash != decrypted_data.get('data_hash'):
            return {
                'status': 'error',
                'message': "Data integrity check failed"
            }
        
        # Verify data integrity using blockchain
        verification = self.blockchain_sharing.verify_data_integrity(
            secure_package['pet_id'], 
            secure_package['data_hash']
        )
        
        if not verification['verified']:
            return {
                'status': 'warning',
                'message': "Data not verified in blockchain",
                'data': decrypted_data
            }
        
        # Log data access
        for category in secure_package['data_categories']:
            self.blockchain_sharing.log_data_access(
                secure_package['pet_id'],
                secure_package['recipient_id'],
                category,
                'import'
            )
        
        logger.info(f"Imported shared data for pet {secure_package['pet_id']}")
        
        return {
            'status': 'success',
            'message': "Data imported successfully",
            'data': decrypted_data,
            'verification': verification
        }
    
    def prepare_data_for_sharing(self, pet_id: str, data: Dict[str, Any],
                                categories: List[str]) -> Dict[str, Any]:
        """
        Prepare data for sharing by adding necessary metadata.
        
        Args:
            pet_id: ID of the pet
            data: Data to prepare for sharing
            categories: Data categories included
            
        Returns:
            Prepared data with metadata
        """
        prepared_data = {
            'pet_id': pet_id,
            'prepared_at': time.time(),
            'data_categories': categories,
            'blockchain_verification': True,
            'data': data
        }
        
        # Generate data hash
        data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        prepared_data['data_hash'] = data_hash
        
        # Register data integrity in blockchain
        self.blockchain_sharing.register_data_integrity(
            pet_id,
            'prepared_shared_data',
            data_hash,
            {
                'categories': categories,
                'prepared_at': prepared_data['prepared_at']
            }
        )
        
        logger.info(f"Prepared data for sharing for pet {pet_id}")
        
        return prepared_data


# Example usage
if __name__ == "__main__":
    # Initialize blockchain data sharing
    blockchain_sharing = BlockchainDataSharing()
    
    # Initialize shared data exchange
    data_exchange = SharedDataExchange(blockchain_sharing)
    
    # Example: Share pet data with a veterinarian
    share_result = blockchain_sharing.share_pet_data(
        "pet12345",  # Pet ID
        "vet98765",  # Veterinarian ID
        ["basic_profile", "medical_records"],  # Data categories
        time.time() + 7 * 24 * 60 * 60  # 7 days expiration
    )
    
    print(f"Shared pet data: {share_result}")
    
    # Example: Validate access permission
    validation_result = blockchain_sharing.validate_access_permission(
        "pet12345",
        "vet98765",
        "medical_records"
    )
    
    print(f"Access validation: {validation_result}")
    
    # Example: Log data access
    blockchain_sharing.log_data_access(
        "pet12345",
        "vet98765",
        "medical_records",
        "read"
    )
    
    # Example: Export shared data
    export_result = data_exchange.export_shared_data(
        "pet12345",
        "vet98765",
        ["basic_profile", "medical_records"]
    )
    
    print(f"Exported data: {export_result}")
