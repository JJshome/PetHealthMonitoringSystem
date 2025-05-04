"""
Blockchain Data Sharing Example

This script demonstrates the functionality of the blockchain-based data sharing module
in the Pet Health Monitoring System.

Technical Note: This implementation is based on patented technology filed by Ucaretron Inc.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
import logging

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the data management module
from data_management_module import DataStorageManager, BlockchainDataSharing, SharedDataExchange

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('blockchain_example')

def main():
    """Main function demonstrating blockchain data sharing functionality."""
    
    # Set up data directories
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    blockchain_dir = os.path.join(data_dir, 'blockchain_data')
    
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(blockchain_dir, exist_ok=True)
    
    # Initialize the data storage manager
    logger.info("Initializing data storage manager...")
    storage_manager = DataStorageManager(data_dir)
    
    # Initialize blockchain data sharing
    logger.info("Initializing blockchain data sharing...")
    blockchain_sharing = BlockchainDataSharing(blockchain_dir)
    
    # Initialize shared data exchange
    logger.info("Initializing shared data exchange...")
    data_exchange = SharedDataExchange(blockchain_sharing)
    
    # Create example pet profiles
    create_example_pet_profiles(storage_manager)
    
    # Demo sharing data between pet owner and veterinarian
    demo_blockchain_data_sharing(blockchain_sharing, data_exchange)
    
    # Demo data verification and integrity checks
    demo_data_verification(blockchain_sharing, data_exchange)
    
    logger.info("Blockchain data sharing example completed")

def create_example_pet_profiles(storage_manager):
    """Create example pet profiles for demonstration."""
    
    # Example pet 1: Dog
    dog_info = {
        'name': 'Max',
        'species': 'dog',
        'breed': 'Golden Retriever',
        'age': 3,
        'weight': 28.5,  # kg
        'gender': 'male',
        'color': 'golden',
        'microchip_id': '123456789012345',
        'birth_date': '2022-01-15',
        'neutered': True,
        'owner_info': {
            'name': 'John Smith',
            'phone': '555-123-4567',
            'email': 'john.smith@example.com'
        },
        'vet_info': {
            'name': 'Dr. Wilson',
            'clinic': 'Healthy Pets Clinic',
            'phone': '555-987-6543'
        },
        'allergies': ['Chicken', 'Certain antibiotics'],
        'medications': []
    }
    
    # Example pet 2: Cat
    cat_info = {
        'name': 'Luna',
        'species': 'cat',
        'breed': 'Maine Coon',
        'age': 2,
        'weight': 4.8,  # kg
        'gender': 'female',
        'color': 'black and white',
        'microchip_id': '987654321098765',
        'birth_date': '2023-03-22',
        'neutered': True,
        'owner_info': {
            'name': 'Sarah Johnson',
            'phone': '555-789-1234',
            'email': 'sarah.j@example.com'
        },
        'vet_info': {
            'name': 'Dr. Wilson',
            'clinic': 'Healthy Pets Clinic',
            'phone': '555-987-6543'
        },
        'allergies': [],
        'medications': ['Flea treatment']
    }
    
    # Store pet profiles
    dog_id = "pet_dog_12345"
    cat_id = "pet_cat_67890"
    
    storage_manager.store_pet_profile(dog_id, dog_info)
    storage_manager.store_pet_profile(cat_id, cat_info)
    
    # Store example health records
    dog_health_record = {
        'record_type': 'examination',
        'date': datetime.now().isoformat(),
        'vet': 'Dr. Wilson',
        'clinic': 'Healthy Pets Clinic',
        'weight': 28.7,  # kg
        'temperature': 38.5,  # Celsius
        'heart_rate': 95,  # bpm
        'diagnosis': {
            'issues': [
                {'type': 'allergic_reaction', 'severity': 'mild'},
                {'type': 'ear_infection', 'severity': 'moderate'}
            ]
        },
        'treatment': {
            'medications': [
                {
                    'name': 'Otibiotic Ointment',
                    'dosage': '5 drops per affected ear',
                    'frequency': 'twice daily',
                    'duration': '10 days'
                },
                {
                    'name': 'Allergy Relief',
                    'dosage': '1 tablet (10mg)',
                    'frequency': 'once daily',
                    'duration': '14 days'
                }
            ],
            'instructions': 'Keep ears clean and dry. Return for follow-up in 2 weeks.'
        }
    }
    
    cat_health_record = {
        'record_type': 'examination',
        'date': datetime.now().isoformat(),
        'vet': 'Dr. Wilson',
        'clinic': 'Healthy Pets Clinic',
        'weight': 4.9,  # kg
        'temperature': 38.3,  # Celsius
        'heart_rate': 140,  # bpm
        'diagnosis': {
            'issues': [
                {'type': 'dental_issues', 'severity': 'moderate'}
            ]
        },
        'treatment': {
            'medications': [
                {
                    'name': 'Dental Treats',
                    'dosage': '1 treat',
                    'frequency': 'daily',
                    'duration': 'ongoing'
                }
            ],
            'instructions': 'Schedule dental cleaning in the next 1-2 months.'
        }
    }
    
    storage_manager.store_health_record(dog_id, dog_health_record)
    storage_manager.store_health_record(cat_id, cat_health_record)
    
    # Store example care plans
    dog_care_plan = {
        'plan_type': 'treatment',
        'created_at': datetime.now().isoformat(),
        'valid_until': (datetime.now() + timedelta(days=30)).isoformat(),
        'created_by': 'Dr. Wilson',
        'health_management': {
            'conditions': [
                {
                    'condition': 'ear_infection',
                    'management': 'Apply prescribed medication and keep ears dry'
                },
                {
                    'condition': 'allergic_reaction',
                    'management': 'Avoid chicken in diet, continue allergy medication'
                }
            ]
        },
        'nutrition': {
            'diet_type': 'Hypoallergenic',
            'feeding_schedule': '2 cups, twice daily',
            'supplements': ['Omega-3 fatty acids']
        },
        'activity': {
            'exercise_requirements': 'Moderate - 30-45 minutes of walking daily',
            'restrictions': 'Avoid swimming until ear infection resolves'
        },
        'follow_up': {
            'next_appointment': (datetime.now() + timedelta(days=14)).isoformat(),
            'monitoring': 'Watch for increased scratching or head shaking'
        }
    }
    
    cat_care_plan = {
        'plan_type': 'preventive',
        'created_at': datetime.now().isoformat(),
        'valid_until': (datetime.now() + timedelta(days=180)).isoformat(),
        'created_by': 'Dr. Wilson',
        'health_management': {
            'conditions': [
                {
                    'condition': 'dental_issues',
                    'management': 'Daily dental treats, schedule professional cleaning'
                }
            ]
        },
        'nutrition': {
            'diet_type': 'Dental Formula',
            'feeding_schedule': '1/2 cup, three times daily',
            'supplements': ['Dental water additive']
        },
        'activity': {
            'exercise_requirements': 'Interactive play for 15-20 minutes, twice daily',
            'restrictions': 'None'
        },
        'follow_up': {
            'next_appointment': (datetime.now() + timedelta(days=60)).isoformat(),
            'monitoring': 'Watch for signs of dental discomfort'
        }
    }
    
    storage_manager.store_care_plan(dog_id, dog_care_plan)
    storage_manager.store_care_plan(cat_id, cat_care_plan)
    
    logger.info(f"Created example profiles for pets: {dog_id} and {cat_id}")
    
    return dog_id, cat_id

def demo_blockchain_data_sharing(blockchain_sharing, data_exchange):
    """Demonstrate blockchain data sharing functionality."""
    
    logger.info("\n===== BLOCKCHAIN DATA SHARING DEMO =====")
    
    # Define pet and user IDs for demo
    pet_id = "pet_dog_12345"
    owner_id = "owner_12345"
    vet_id = "vet_98765"
    researcher_id = "researcher_54321"
    
    # 1. Share pet data with veterinarian
    logger.info("\n1. Sharing pet data with veterinarian...")
    vet_share = blockchain_sharing.share_pet_data(
        pet_id=pet_id,
        recipient_id=vet_id,
        data_categories=["basic_profile", "medical_records", "sensor_data"],
        expiration_time=int(time.time() + 30 * 24 * 60 * 60),  # 30 days
        access_level="read",
        custom_message="Sharing for regular check-up and ongoing treatment"
    )
    logger.info(f"  - Share ID: {vet_share['share_id']}")
    logger.info(f"  - Expires: {datetime.fromtimestamp(vet_share['expires_at']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 2. Share limited data with researcher
    logger.info("\n2. Sharing limited pet data with researcher...")
    researcher_share = blockchain_sharing.share_pet_data(
        pet_id=pet_id,
        recipient_id=researcher_id,
        data_categories=["anonymized_medical_data"],
        expiration_time=int(time.time() + 180 * 24 * 60 * 60),  # 180 days
        access_level="read",
        custom_message="Sharing for research study on pet allergies"
    )
    logger.info(f"  - Share ID: {researcher_share['share_id']}")
    logger.info(f"  - Expires: {datetime.fromtimestamp(researcher_share['expires_at']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 3. Validate access permissions
    logger.info("\n3. Validating access permissions...")
    
    vet_permission = blockchain_sharing.validate_access_permission(
        pet_id, vet_id, "medical_records"
    )
    logger.info(f"  - Vet has access to medical records: {vet_permission['has_access']}")
    
    researcher_permission = blockchain_sharing.validate_access_permission(
        pet_id, researcher_id, "medical_records"
    )
    logger.info(f"  - Researcher has access to medical records: {researcher_permission['has_access']}")
    
    researcher_anon_permission = blockchain_sharing.validate_access_permission(
        pet_id, researcher_id, "anonymized_medical_data"
    )
    logger.info(f"  - Researcher has access to anonymized data: {researcher_anon_permission['has_access']}")
    
    # 4. Log data access
    logger.info("\n4. Logging data access events...")
    
    blockchain_sharing.log_data_access(
        pet_id, vet_id, "medical_records", "read", "success"
    )
    logger.info("  - Logged vet's access to medical records")
    
    blockchain_sharing.log_data_access(
        pet_id, researcher_id, "anonymized_medical_data", "read", "success"
    )
    logger.info("  - Logged researcher's access to anonymized data")
    
    # 5. Export shared data
    logger.info("\n5. Exporting shared data for veterinarian...")
    
    export_result = data_exchange.export_shared_data(
        pet_id, vet_id, ["basic_profile", "medical_records"]
    )
    logger.info(f"  - Export status: {export_result['status']}")
    logger.info(f"  - Verification code: {export_result['otp']}")
    
    # 6. Revoke share with researcher
    logger.info("\n6. Revoking researcher access...")
    
    revoke_result = blockchain_sharing.revoke_shared_data(
        researcher_share['share_id'],
        "Study completed"
    )
    logger.info(f"  - Revocation status: {revoke_result['status']}")
    logger.info(f"  - Revocation time: {datetime.fromtimestamp(revoke_result['revoked_at']).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 7. Validate revoked access
    logger.info("\n7. Validating revoked access...")
    
    researcher_revoked = blockchain_sharing.validate_access_permission(
        pet_id, researcher_id, "anonymized_medical_data"
    )
    logger.info(f"  - Researcher still has access: {researcher_revoked['has_access']}")
    
    # 8. Get all active shares
    logger.info("\n8. Getting all active shares...")
    
    active_shares = blockchain_sharing.get_active_shares_for_pet(pet_id)
    logger.info(f"  - Active shares count: {len(active_shares)}")
    for share in active_shares:
        logger.info(f"  - Share ID: {share['share_id']}, Recipient: {share['recipient_id']}")

def demo_data_verification(blockchain_sharing, data_exchange):
    """Demonstrate data verification and integrity functionality."""
    
    logger.info("\n===== DATA VERIFICATION DEMO =====")
    
    # Define pet ID for demo
    pet_id = "pet_dog_12345"
    
    # 1. Register data integrity hash
    logger.info("\n1. Registering data integrity hash...")
    
    # Sample data
    sample_data = {
        'pet_id': pet_id,
        'data_type': 'health_report',
        'timestamp': time.time(),
        'content': {
            'weight': 28.9,
            'temperature': 38.2,
            'heart_rate': 90,
            'blood_pressure': '120/80',
            'general_condition': 'good'
        }
    }
    
    # Calculate hash
    data_hash = hashlib.sha256(json.dumps(sample_data, sort_keys=True).encode()).hexdigest()
    
    # Register in blockchain
    register_result = blockchain_sharing.register_data_integrity(
        pet_id,
        'health_report',
        data_hash,
        {'report_id': 'HR-2023-001', 'version': '1.0'}
    )
    
    logger.info(f"  - Registration status: {register_result['status']}")
    logger.info(f"  - Block index: {register_result['block_index']}")
    logger.info(f"  - Data hash: {data_hash}")
    
    # 2. Verify data integrity
    logger.info("\n2. Verifying data integrity...")
    
    verify_result = blockchain_sharing.verify_data_integrity(pet_id, data_hash)
    logger.info(f"  - Verification status: {verify_result['status']}")
    logger.info(f"  - Verified: {verify_result['verified']}")
    
    if verify_result['verified']:
        logger.info(f"  - Recorded at: {datetime.fromtimestamp(verify_result['recorded_at']).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  - Block index: {verify_result['block_index']}")
    
    # 3. Verify tampered data
    logger.info("\n3. Verifying tampered data...")
    
    # Tampered data
    tampered_data = sample_data.copy()
    tampered_data['content']['weight'] = 30.1  # Changed weight
    
    # Calculate new hash
    tampered_hash = hashlib.sha256(json.dumps(tampered_data, sort_keys=True).encode()).hexdigest()
    
    # Verify
    tampered_verify = blockchain_sharing.verify_data_integrity(pet_id, tampered_hash)
    logger.info(f"  - Verification status: {tampered_verify['status']}")
    logger.info(f"  - Verified: {tampered_verify['verified']}")
    logger.info(f"  - Original hash: {data_hash}")
    logger.info(f"  - Tampered hash: {tampered_hash}")
    
    # 4. Prepare data for sharing
    logger.info("\n4. Preparing data for sharing with verification...")
    
    prepared_data = data_exchange.prepare_data_for_sharing(
        pet_id,
        sample_data['content'],
        ['health_report']
    )
    
    logger.info(f"  - Prepared data hash: {prepared_data['data_hash']}")
    logger.info(f"  - Blockchain verification: {prepared_data['blockchain_verification']}")
    
    # 5. Get access logs
    logger.info("\n5. Getting access logs...")
    
    access_logs = blockchain_sharing.get_access_logs_for_pet(pet_id, limit=5)
    logger.info(f"  - Access logs count: {len(access_logs)}")
    
    for i, log in enumerate(access_logs[:3]):  # Show first 3 logs
        logger.info(f"  - Log {i+1}: Type: {log.get('type')}, Time: {datetime.fromtimestamp(log.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
