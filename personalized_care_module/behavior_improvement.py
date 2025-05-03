"""
Behavior Improvement Module

This module provides specialized behavior analysis and improvement plans for pets with
behavioral issues. It works in conjunction with the Care Plan Generator to create
customized training protocols and environmental modifications.

Technical Note: This implementation is based on patented technology filed by Ucaretron Inc.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import random

class BehaviorImprovementSystem:
    """
    Specialized system for analyzing and addressing pet behavioral issues.
    
    This class provides detailed analysis of behavior problems and generates
    customized improvement plans with specific training protocols, 
    environmental modifications, and enrichment recommendations.
    """
    
    def __init__(self):
        """Initialize the behavior improvement system"""
        # Behavior categories and associated techniques
        self.behavior_techniques = {
            'anxiety': {
                'core_techniques': [
                    'Systematic desensitization',
                    'Counter-conditioning',
                    'Relaxation protocol',
                    'Safe space creation'
                ],
                'secondary_techniques': [
                    'Environmental enrichment',
                    'Predictable routines',
                    'Calming signals',
                    'Aromatherapy',
                    'Anxiety wraps',
                    'Classical music'
                ]
            },
            'aggression': {
                'core_techniques': [
                    'Trigger identification and management',
                    'Response substitution',
                    'Impulse control training',
                    'Structured management programs'
                ],
                'secondary_techniques': [
                    'Look Away protocol',
                    'Emergency responses',
                    'Safety routines',
                    'Avoidance training',
                    'Give-way protocols'
                ]
            },
            'separation_anxiety': {
                'core_techniques': [
                    'Graduated absence training',
                    'Departure cue desensitization',
                    'Independence reinforcement',
                    'Self-calming behavior training'
                ],
                'secondary_techniques': [
                    'Pre-departure routine adjustment',
                    'Relaxation protocols',
                    'Audio/visual stimulation',
                    'Comfort object preparation',
                    'Autonomy exercises'
                ]
            },
            'destructive_behavior': {
                'core_techniques': [
                    'Appropriate outlet provision',
                    'Incompatible behavior training',
                    'Management and supervision',
                    'Enrichment scheduling'
                ],
                'secondary_techniques': [
                    'Taste aversion conditioning',
                    'Texture preference analysis',
                    'Chew toy rotation',
                    'Interactive play scheduling',
                    'Frustration tolerance building'
                ]
            }
        }
        
        # Training protocols for common behavior issues
        self.training_protocols = {
            'anxiety': [
                {
                    'name': 'Gradual exposure protocol',
                    'difficulty': 'moderate',
                    'duration': '8-12 weeks',
                    'steps': [
                        'Identify specific anxiety triggers and their intensity',
                        'Establish baseline reactions and distance thresholds',
                        'Create positive associations at sub-threshold exposures',
                        'Gradually increase exposure while maintaining positive emotional state',
                        'Practice relaxation exercises during controlled exposures',
                        'Generalize exposure work to different environments'
                    ],
                    'success_metrics': [
                        'Decreased physiological signs of stress',
                        'Voluntary approach behavior',
                        'Faster recovery after exposure',
                        'Relaxed body language during exposure'
                    ]
                },
                {
                    'name': 'Safe space training',
                    'difficulty': 'easy',
                    'duration': '2-4 weeks',
                    'steps': [
                        'Identify ideal location(s) for safe spaces',
                        'Create comfortable, enclosed retreats in key areas',
                        'Positively condition pet to each safe space',
                        'Pair safe spaces with high-value rewards',
                        'Teach "go to place" cue for each safe space',
                        'Practice sending to safe spaces during mild stress'
                    ],
                    'success_metrics': [
                        'Voluntary use of safe spaces when stressed',
                        'Reduced anxiety behaviors when in safe space',
                        'Reliable response to "go to place" cue',
                        'Decreased overall household anxiety'
                    ]
                }
            ],
            'aggression': [
                {
                    'name': 'Trigger management protocol',
                    'difficulty': 'high',
                    'duration': '12-16 weeks',
                    'steps': [
                        'Conduct comprehensive trigger assessment',
                        'Implement management system to prevent rehearsal of aggression',
                        'Establish baseline responses at safe distances',
                        'Teach alternative behaviors to replace aggressive responses',
                        'Create gradual exposure plan with positive reinforcement',
                        'Implement systematic counter-conditioning for each trigger'
                    ],
                    'success_metrics': [
                        'Decreased frequency of aggressive displays',
                        'Increased warning signals before reaction',
                        'Improved response to redirect cues',
                        'Successful performance of alternative behaviors'
                    ]
                }
            ],
            'separation_anxiety': [
                {
                    'name': 'Graduated absence protocol',
                    'difficulty': 'high',
                    'duration': '8-16 weeks',
                    'steps': [
                        'Establish baseline anxiety threshold (duration before distress)',
                        'Implement complete management to prevent anxiety spikes',
                        'Desensitize to pre-departure cues separately',
                        'Practice sub-threshold departures with gradual duration increases',
                        'Introduce departure enrichment activities',
                        'Systematically vary departure routines and durations'
                    ],
                    'success_metrics': [
                        'Calm behavior during pre-departure cues',
                        'Engagement with activities when alone',
                        'Absence of distress vocalizations',
                        'Relaxed greeting upon return'
                    ]
                }
            ]
        }
        
        # Environmental modification guides
        self.environmental_modifications = {
            'anxiety': {
                'home': [
                    'Create covered, den-like resting areas in low-traffic locations',
                    'Establish consistent daily routines for all activities',
                    'Provide white noise or calming music in rest areas',
                    'Use pheromone diffusers in key areas',
                    'Control access to windows/doors for reactive pets',
                    'Create visual barriers for stress-inducing views'
                ],
                'products': [
                    'Covered crates or beds',
                    'White noise machines',
                    'Pheromone diffusers (Adaptil/Feliway)',
                    'Anxiety wraps (Thundershirt)',
                    'Calming supplements (as recommended by veterinarian)',
                    'Calming caps or masks for visual stimulation reduction'
                ]
            },
            'aggression': {
                'home': [
                    'Install secure physical barriers between trigger sources',
                    'Create separate spaces for resource access (food, beds, toys)',
                    'Implement visual barriers where needed (frosted film, barriers)',
                    'Establish structured traffic patterns to avoid conflict zones',
                    'Remove environmental triggers when possible',
                    'Create safe spaces for each pet away from others'
                ],
                'products': [
                    'Baby gates or exercise pens',
                    'Properly fitted basket muzzles',
                    'Front-clip harnesses or head halters',
                    'Barrier systems (ex-pens, room dividers)',
                    'Safety warning gear (yellow ribbon, reactive dog vest)',
                    'Long-lines for distance control'
                ]
            },
            'separation_anxiety': {
                'home': [
                    'Create comfortable, secure resting area',
                    'Provide engaging alone-time activities',
                    'Use background noise (radio, TV) for auditory masking',
                    'Block visual access to departure areas when alone',
                    'Provide access to familiar scent items (owner's clothing)',
                    'Install pet camera for monitoring'
                ],
                'products': [
                    'Long-lasting food puzzles',
                    'Automatic treat dispensers',
                    'Interactive toys',
                    'Comfort items with owner's scent',
                    'White noise machine',
                    'Pheromone diffusers'
                ]
            }
        }
        
        # Enrichment recommendations
        self.enrichment_activities = {
            'intellectual': [
                {
                    'name': 'Food puzzle rotation',
                    'benefits': ['Mental stimulation', 'Problem-solving skills', 'Slow eating'],
                    'species': ['dog', 'cat'],
                    'implementation': 'Rotate 3-5 different food puzzles of varying difficulties.'
                },
                {
                    'name': 'Scent work games',
                    'benefits': ['Mental fatigue', 'Confidence building', 'Natural behavior outlet'],
                    'species': ['dog'],
                    'implementation': 'Hide treats or toys around home at increasing levels of difficulty.'
                }
            ],
            'physical': [
                {
                    'name': 'Flirt pole sessions',
                    'benefits': ['High-intensity exercise', 'Prey drive outlet', 'Impulse control'],
                    'species': ['dog', 'cat'],
                    'implementation': 'Use 5-10 minute structured sessions with rules.'
                },
                {
                    'name': 'Fetch variations',
                    'benefits': ['Aerobic exercise', 'Impulse control', 'Human-animal bonding'],
                    'species': ['dog'],
                    'implementation': 'Teach structured retrieve with clear start/stop cues.'
                }
            ],
            'environmental': [
                {
                    'name': 'Rotation toy system',
                    'benefits': ['Maintained interest', 'Reduced habituation', 'Varied stimulation'],
                    'species': ['dog', 'cat'],
                    'implementation': 'Divide toys into 3-4 sets. Rotate sets weekly to maintain novelty.'
                }
            ]
        }
        
        print("Behavior Improvement System initialized")
    
    def analyze_behavior_issue(self, issue_data: Dict[str, Any], pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a specific behavioral issue and generate a comprehensive improvement plan.
        
        Args:
            issue_data: Data describing the behavioral issue
            pet_info: Information about the pet (species, breed, age, etc.)
            
        Returns:
            Comprehensive behavior analysis and improvement plan
        """
        # Extract key information
        issue_type = issue_data.get('type', '').lower()
        severity = issue_data.get('severity', 'moderate').lower()
        
        # Identify behavior category
        category = self._determine_behavior_category(issue_type)
        
        # Create base analysis structure
        analysis = {
            'issue': issue_data.get('type'),
            'category': category,
            'severity': severity,
            'triggers': self._identify_potential_triggers(issue_type, pet_info),
            'contributing_factors': self._identify_contributing_factors(issue_type, pet_info),
            'improvement_plan': {
                'recommended_approach': '',
                'core_techniques': [],
                'training_protocols': [],
                'environmental_modifications': [],
                'enrichment_activities': [],
                'expected_timeline': '',
                'success_indicators': [],
                'professional_guidance': severity == 'severe'
            }
        }
        
        # Set approach based on category and severity
        if category:
            approach = f"{severity} {category} behavior modification protocol"
        else:
            approach = f"{severity} behavior modification protocol"
        analysis['improvement_plan']['recommended_approach'] = approach
        
        # Add recommended techniques
        if category and category in self.behavior_techniques:
            # Add core techniques
            analysis['improvement_plan']['core_techniques'] = self.behavior_techniques[category]['core_techniques'][:3]
            
            # Add secondary techniques based on severity
            if severity == 'severe':
                analysis['improvement_plan']['core_techniques'].extend(
                    self.behavior_techniques[category]['secondary_techniques'][:2])
            elif severity == 'moderate':
                analysis['improvement_plan']['core_techniques'].append(
                    self.behavior_techniques[category]['secondary_techniques'][0])
        
        # Add training protocols
        if category and category in self.training_protocols:
            protocols = self.training_protocols[category]
            
            # Select appropriate protocols based on severity
            if severity == 'severe':
                analysis['improvement_plan']['training_protocols'] = protocols
            elif severity == 'moderate':
                analysis['improvement_plan']['training_protocols'] = protocols[:1]
            else:  # mild
                if protocols:
                    # Simplify the protocol for mild cases
                    simplified_protocol = protocols[0].copy()
                    simplified_protocol['steps'] = simplified_protocol['steps'][:4]
                    simplified_protocol['duration'] = '4-6 weeks'
                    analysis['improvement_plan']['training_protocols'] = [simplified_protocol]
        
        # Add environmental modifications
        if category and category in self.environmental_modifications:
            mods = self.environmental_modifications[category]
            
            # Add home modifications
            analysis['improvement_plan']['environmental_modifications'] = [
                {'area': 'home', 'modifications': mods['home'][:3]}
            ]
            
            # Add product recommendations
            analysis['improvement_plan']['environmental_modifications'].append(
                {'area': 'products', 'recommendations': mods['products'][:3]}
            )
        
        # Add enrichment activities
        analysis['improvement_plan']['enrichment_activities'] = self._select_enrichment_activities(category, pet_info)
        
        # Set expected timeline based on severity
        timelines = {
            'severe': '4-6 months with consistent implementation',
            'moderate': '2-3 months with consistent implementation',
            'mild': '4-6 weeks with consistent implementation'
        }
        analysis['improvement_plan']['expected_timeline'] = timelines.get(severity, '2-3 months')
        
        # Set success indicators
        analysis['improvement_plan']['success_indicators'] = self._generate_success_indicators(issue_type, severity)
        
        # Add professional guidance recommendations for severe cases
        if severity == 'severe':
            analysis['improvement_plan']['professional_guidance_recommendations'] = [
                'Consult with a certified animal behaviorist or veterinary behaviorist',
                'Consider behavior modification medication in conjunction with training',
                'Work with a certified professional dog trainer for implementation support',
                'Regular progress monitoring and plan adjustment'
            ]
        
        return analysis
    
    def _determine_behavior_category(self, issue_type: str) -> str:
        """
        Determine the behavior category based on the issue type.
        
        Args:
            issue_type: Description of the behavioral issue
            
        Returns:
            Behavior category
        """
        issue_type = issue_type.lower()
        
        # Direct matches
        if 'anxiety' in issue_type:
            return 'anxiety'
        elif 'aggression' in issue_type or 'aggressive' in issue_type or 'biting' in issue_type:
            return 'aggression'
        elif 'separation' in issue_type:
            return 'separation_anxiety'
        elif 'destructive' in issue_type or 'chew' in issue_type or 'scratch' in issue_type:
            return 'destructive_behavior'
        
        # More general matches
        keywords = {
            'anxiety': ['nervous', 'stress', 'panic', 'worry', 'timid', 'fear', 'phobia'],
            'aggression': ['growl', 'snap', 'lunge', 'threat', 'fight', 'guard', 'reactive'],
            'separation_anxiety': ['alone', 'abandon', 'isolation', 'panic when left'],
            'destructive_behavior': ['destroy', 'rip', 'tear', 'damage', 'inappropriate chewing']
        }
        
        for category, words in keywords.items():
            if any(word in issue_type for word in words):
                return category
        
        # Default fallback
        return 'general_behavior'
    
    def _identify_potential_triggers(self, issue_type: str, pet_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify potential triggers for the behavioral issue.
        
        Args:
            issue_type: Description of the behavioral issue
            pet_info: Information about the pet
            
        Returns:
            List of potential triggers
        """
        issue_type = issue_type.lower()
        species = pet_info.get('species', '').lower()
        
        triggers = []
        
        # Anxiety triggers
        if 'anxiety' in issue_type or 'fear' in issue_type or 'phobia' in issue_type:
            triggers = [
                {'trigger': 'Loud noises', 'management': 'Sound reduction, safe space access'},
                {'trigger': 'Novel environments', 'management': 'Gradual exposure, familiar items'}
            ]
            
            if species == 'dog':
                triggers.extend([
                    {'trigger': 'Unfamiliar dogs', 'management': 'Controlled distance, positive association'},
                    {'trigger': 'Crowded spaces', 'management': 'Avoidance, gradual desensitization'}
                ])
            elif species == 'cat':
                triggers.extend([
                    {'trigger': 'Environmental changes', 'management': 'Gradual transitions, familiar scents'},
                    {'trigger': 'Confined spaces', 'management': 'Alternative escape routes, hiding options'}
                ])
        
        # Aggression triggers
        elif 'aggression' in issue_type or 'aggressive' in issue_type:
            triggers = [
                {'trigger': 'Resource guarding', 'management': 'Separate feeding, toy rotation'},
                {'trigger': 'Territory defense', 'management': 'Controlled access, visual barriers'}
            ]
            
            if species == 'dog':
                triggers.extend([
                    {'trigger': 'Fear of handling', 'management': 'Consent-based handling, cooperative care'},
                    {'trigger': 'Reactivity to other dogs', 'management': 'Leash control, distance management'}
                ])
            elif species == 'cat':
                triggers.extend([
                    {'trigger': 'Petting aggression', 'management': 'Recognize signs, respect preferences'},
                    {'trigger': 'Inter-cat conflict', 'management': 'Resource separation, reintroduction protocol'}
                ])
        
        # Separation anxiety triggers
        elif 'separation' in issue_type:
            triggers = [
                {'trigger': 'Owner departure cues', 'management': 'Desensitize to departure signals'},
                {'trigger': 'Being left alone', 'management': 'Graduated absences, safe confinement'},
                {'trigger': 'Sudden schedule changes', 'management': 'Consistent routine, gradual transitions'},
                {'trigger': 'Isolation from social group', 'management': 'Companion presence when possible'}
            ]
        
        # Destructive behavior triggers
        elif 'destructive' in issue_type or 'chew' in issue_type or 'scratch' in issue_type:
            triggers = [
                {'trigger': 'Boredom', 'management': 'Mental stimulation, appropriate chew items'},
                {'trigger': 'Excess energy', 'management': 'Increased exercise, structured play'},
                {'trigger': 'Anxiety', 'management': 'Identify underlying anxiety source, address root cause'},
                {'trigger': 'Teething (young pets)', 'management': 'Appropriate chew toys, frozen items for pain'}
            ]
        
        return triggers
    
    def _identify_contributing_factors(self, issue_type: str, pet_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Identify potential contributing factors to the behavioral issue.
        
        Args:
            issue_type: Description of the behavioral issue
            pet_info: Information about the pet
            
        Returns:
            List of potential contributing factors
        """
        issue_type = issue_type.lower()
        species = pet_info.get('species', '').lower()
        breed = pet_info.get('breed', '').lower()
        age = pet_info.get('age', 0)
        
        factors = []
        
        # Add general factors based on issue type
        if 'anxiety' in issue_type or 'fear' in issue_type:
            factors.extend([
                {'factor': 'Insufficient socialization', 'impact': 'Increased fear response to novelty'},
                {'factor': 'Previous negative experiences', 'impact': 'Learned fear associations'},
                {'factor': 'Genetic predisposition', 'impact': 'Heightened stress response'}
            ])
        
        elif 'aggression' in issue_type:
            factors.extend([
                {'factor': 'Insufficient socialization', 'impact': 'Poor social skills development'},
                {'factor': 'Resource insecurity', 'impact': 'Increased guarding behavior'},
                {'factor': 'Pain or discomfort', 'impact': 'Defensive aggression when handled'}
            ])
        
        elif 'separation' in issue_type:
            factors.extend([
                {'factor': 'Hyper-attachment', 'impact': 'Inability to cope with separation'},
                {'factor': 'Insufficient independence training', 'impact': 'Lack of coping skills'},
                {'factor': 'Sudden schedule changes', 'impact': 'Unpredictability causing anxiety'}
            ])
        
        elif 'destructive' in issue_type:
            factors.extend([
                {'factor': 'Insufficient physical exercise', 'impact': 'Excess energy expressed destructively'},
                {'factor': 'Insufficient mental stimulation', 'impact': 'Boredom-based destruction'},
                {'factor': 'Chewing/scratching needs unmet', 'impact': 'Natural behaviors without appropriate outlets'}
            ])
        
        # Add species-specific factors
        if species == 'dog':
            # Add breed-specific factors for dogs
            high_energy_breeds = ['labrador', 'shepherd', 'terrier', 'husky', 'collie', 'retriever']
            guarding_breeds = ['shepherd', 'rottweiler', 'doberman', 'mastiff']
            
            if any(b in breed for b in high_energy_breeds):
                factors.append({
                    'factor': 'High-energy breed genetics', 
                    'impact': 'Increased exercise and stimulation requirements'
                })
            
            if any(b in breed for b in guarding_breeds) and 'aggression' in issue_type:
                factors.append({
                    'factor': 'Breed-specific guarding tendencies', 
                    'impact': 'More pronounced territorial/protective behaviors'
                })
        
        elif species == 'cat':
            factors.extend([
                {'factor': 'Environmental restrictions', 'impact': 'Limited territory access increasing stress'},
                {'factor': 'Multi-cat tensions', 'impact': 'Social stress from forced proximity'}
            ])
        
        # Add age-related factors
        if age < 2:
            factors.append({
                'factor': 'Developmental phase', 
                'impact': 'Age-appropriate behavior testing and exploration'
            })
        elif age > 8:
            factors.append({
                'factor': 'Age-related changes', 
                'impact': 'Potential cognitive changes or increased sensitivity'
            })
        
        return factors
    
    def _select_enrichment_activities(self, category: str, pet_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select appropriate enrichment activities based on behavioral category and pet information.
        
        Args:
            category: Behavioral category
            pet_info: Information about the pet
            
        Returns:
            List of enrichment activities
        """
        species = pet_info.get('species', '').lower()
        
        selected_activities = []
        
        # Filter by species compatibility
        for category_type, activities in self.enrichment_activities.items():
            for activity in activities:
                if species in activity['species']:
                    # Clone the activity and add the category
                    act_copy = activity.copy()
                    act_copy['category'] = category_type
                    selected_activities.append(act_copy)
        
        # If we have more than 3 activities, select the most appropriate ones
        if len(selected_activities) > 3:
            if category == 'anxiety' or category == 'separation_anxiety':
                # For anxiety, prioritize mental stimulation and environmental enrichment
                priority_categories = ['intellectual', 'environmental']
            elif category == 'aggression':
                # For aggression, prioritize physical and mental outlets
                priority_categories = ['physical', 'intellectual']
            elif category == 'destructive_behavior':
                # For destructive behavior, prioritize alternatives for chewing/scratching
                priority_categories = ['physical', 'environmental']
            else:
                # Default priority
                priority_categories = ['intellectual', 'physical', 'environmental']
            
            # Sort activities by priority
            sorted_activities = []
            for priority in priority_categories:
                for activity in selected_activities:
                    if activity['category'] == priority and activity not in sorted_activities:
                        sorted_activities.append(activity)
            
            # Add any remaining activities
            for activity in selected_activities:
                if activity not in sorted_activities:
                    sorted_activities.append(activity)
            
            # Take the top 3
            selected_activities = sorted_activities[:3]
        
        return selected_activities
    
    def _generate_success_indicators(self, issue_type: str, severity: str) -> List[str]:
        """
        Generate success indicators for the improvement plan.
        
        Args:
            issue_type: Description of the behavioral issue
            severity: Severity of the issue
            
        Returns:
            List of success indicators
        """
        issue_type = issue_type.lower()
        
        # Common success indicators for all behaviors
        common_indicators = [
            'Decreased frequency and intensity of problematic behavior',
            'Improved response to management techniques',
            'Increased display of alternative, appropriate behaviors'
        ]
        
        # Issue-specific indicators
        specific_indicators = []
        
        if 'anxiety' in issue_type or 'fear' in issue_type:
            specific_indicators = [
                'Reduced physiological signs of stress (panting, pacing, drooling)',
                'Voluntary approach to previously feared stimuli',
                'More rapid recovery after exposure to triggers',
                'Decreased startle response to environmental changes'
            ]
        
        elif 'aggression' in issue_type:
            specific_indicators = [
                'Increased warning signals before reaction (improved bite inhibition)',
                'Greater tolerance of trigger presence at safe distances',
                'Successful performance of alternative behaviors when triggered',
                'Reduced intensity of aggressive displays'
            ]
        
        elif 'separation' in issue_type:
            specific_indicators = [
                'Calm behavior during pre-departure routines',
                'Engagement with enrichment while alone',
                'Decreased destructive/vocal behavior in video monitoring',
                'Relaxed body language upon return'
            ]
        
        elif 'destructive' in issue_type:
            specific_indicators = [
                'Redirection of chewing/scratching to appropriate outlets',
                'Decreased damage to household items',
                'Engagement with provided appropriate alternatives',
                'Reduced frustration behaviors'
            ]
        
        # Combine and limit indicators based on severity
        if severity == 'severe':
            return common_indicators + specific_indicators[:3]
        elif severity == 'moderate':
            return common_indicators + specific_indicators[:2]
        else:  # mild
            return common_indicators + specific_indicators[:1]
    
    def generate_training_protocol(self, behavior_category: str, pet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed training protocol for a specific behavior issue.
        
        Args:
            behavior_category: Category of the behavior issue
            pet_info: Information about the pet
            
        Returns:
            Detailed training protocol
        """
        species = pet_info.get('species', '').lower()
        
        # Select a base protocol from the available protocols for this category
        if behavior_category in self.training_protocols and self.training_protocols[behavior_category]:
            # Get the first protocol as a template
            base_protocol = self.training_protocols[behavior_category][0].copy()
        else:
            # Create a generic protocol if none exists for this category
            base_protocol = {
                'name': f"{behavior_category.replace('_', ' ').title()} Improvement Protocol",
                'difficulty': 'moderate',
                'duration': '6-8 weeks',
                'steps': [
                    'Identify specific triggers and contexts',
                    'Implement management to prevent rehearsal of the behavior',
                    'Train foundation behaviors for later use',
                    'Begin systematic desensitization to triggers',
                    'Counter-condition emotional response',
                    'Practice alternative behaviors in progressively challenging contexts'
                ],
                'success_metrics': [
                    'Decreased frequency and intensity of problematic behavior',
                    'Improved response to cues in trigger contexts',
                    'Increased display of alternative behaviors'
                ]
            }
        
        # Customize the protocol based on pet information
        protocol = base_protocol.copy()
        
        # Adjust protocol based on species
        if species == 'cat':
            # Modify for cats - shorter sessions, more environmental focus
            protocol['steps'] = [step.replace('walks', 'play sessions') for step in protocol['steps']]
            protocol['implementation_notes'] = [
                'Keep training sessions to 2-3 minutes maximum',
                'Focus on natural behaviors and positive reinforcement',
                'Environmental management is especially important',
                'Respect cat\'s choice to participate or disengage',
                'Use play and high-value treats as primary reinforcers'
            ]
        else:  # dog
            protocol['implementation_notes'] = [
                'Keep training sessions to 5-10 minutes',
                'Use high-value rewards for best results',
                'Train in low-distraction environments first',
                'Gradually increase difficulty as skills improve',
                'Consistency across all family members is essential'
            ]
        
        # Add session structure
        protocol['session_structure'] = {
            'frequency': '1-2 times daily',
            'duration': '5-10 minutes for dogs, 2-5 minutes for cats',
            'format': [
                'Brief warm-up with familiar skills (1-2 minutes)',
                'Focused work on new skill or behavior (3-5 minutes)',
                'Fun, successful ending exercise (1-2 minutes)'
            ]
        }
        
        # Add troubleshooting guidance
        protocol['troubleshooting'] = [
            {
                'problem': 'Pet shows increased anxiety during training',
                'solution': 'Reduce difficulty, ensure success, use higher value rewards'
            },
            {
                'problem': 'Progress plateau after initial improvement',
                'solution': 'Evaluate criteria, break into smaller steps, vary training contexts'
            },
            {
                'problem': 'Regression in previously improved behavior',
                'solution': 'Return to easier level, reinforce basics, check for environmental changes'
            }
        ]
        
        return protocol


# Example usage
if __name__ == "__main__":
    # Create a behavior improvement system
    behavior_system = BehaviorImprovementSystem()
    
    # Example behavior issue
    anxiety_issue = {
        'type': 'separation anxiety',
        'severity': 'moderate',
        'confidence': 0.85
    }
    
    # Example pet information
    pet_info = {
        'name': 'Max',
        'species': 'dog',
        'breed': 'Golden Retriever',
        'age': 3,
        'weight': 28.5  # kg
    }
    
    # Analyze behavior issue
    analysis = behavior_system.analyze_behavior_issue(anxiety_issue, pet_info)
    
    # Print key recommendations
    print("Behavior Analysis Results:")
    print(f"Issue: {analysis['issue']}")
    print(f"Category: {analysis['category']}")
    print(f"Recommended approach: {analysis['improvement_plan']['recommended_approach']}")
    print("\nCore techniques:")
    for technique in analysis['improvement_plan']['core_techniques']:
        print(f"- {technique}")
        
    print("\nExpected timeline:", analysis['improvement_plan']['expected_timeline'])
